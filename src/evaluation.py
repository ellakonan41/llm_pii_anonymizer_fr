"""Évaluation d'une anonymisation produite par un modèle.

Un exemple contient `source_text`, `masked_text` (référence, étiquettes
dénumérotées) et `privacy_mask` (liste d'entités avec `label`, `value`,
`start`, `end`). La sortie du modèle est le texte anonymisé qu'il a produit.

Quatre mesures :
- protection : part des données personnelles qui ont disparu de la sortie
  (métrique principale, un oubli étant une fuite) ;
- préservation : part des mots hors données personnelles conservés
  (un modèle qui masque tout n'est pas utile) ;
- fidélité : part des textes sans mot ajouté ni étiquette inventée
  (la préservation ne voit pas ce que le modèle ajoute ou modifie) ;
- correspondance exacte avec la référence (indicative, sensible aux
  incohérences d'annotation du jeu de données).
"""

import re
from collections import Counter, defaultdict

from src.prompt import ETIQUETTES

# Longueur minimale d'un fragment de valeur pour être recherché seul
# (évite de compter comme fuite un « 06 » ou un « M » présent ailleurs).
LONGUEUR_MIN_FRAGMENT = 3

# [STREET], [GIVENNAME_1]... : retirées avant la recherche de fuites,
# sinon « Street » serait retrouvé dans l'étiquette [STREET].
_ETIQUETTE = re.compile(r"\[[A-Z0-9_]+\]")


def retirer_etiquettes(texte: str) -> str:
    """Remplace les étiquettes entre crochets par un espace."""
    return _ETIQUETTE.sub(" ", texte)


# Tout contenu entre crochets, pour repérer les étiquettes inventées ([RESOURCES_HUMANITIES], [量]...).
_CROCHETS = re.compile(r"\[([^\[\]]+)\]")


def etiquettes_inventees(sortie: str) -> list[str]:
    """Étiquettes de la sortie qui ne font pas partie des types autorisés."""
    return [e for e in _CROCHETS.findall(sortie) if e not in ETIQUETTES]


def mots_ajoutes(source: str, sortie: str) -> list[str]:
    """Mots de la sortie absents du texte original (hors étiquettes valides).

    Repère les mots inventés ou modifiés par le modèle : « 5 jours » réécrit
    en « [AGE] ans » ajoute « ans », une faute de frappe ajoute un mot inconnu.
    """
    sortie_sans_etiquettes = _CROCHETS.sub(
        lambda m: " " if m.group(1) in ETIQUETTES else m.group(0), sortie
    )
    ajoutes = Counter(re.findall(r"\w+", sortie_sans_etiquettes.lower())) - Counter(
        re.findall(r"\w+", source.lower())
    )
    return sorted(ajoutes.elements())


def valeur_presente(valeur: str, texte: str) -> bool:
    """Indique si `valeur` apparaît dans `texte` comme mot(s) entier(s), sans tenir compte de la casse."""
    motif = rf"(?<!\w){re.escape(valeur)}(?!\w)"
    return re.search(motif, texte, flags=re.IGNORECASE) is not None


def entite_protegee(valeur: str, sortie: str, mots_du_texte: frozenset = frozenset()) -> bool:
    """Indique si une donnée personnelle a entièrement disparu de la sortie.

    La valeur complète ne doit plus apparaître, ni aucun de ses fragments
    suffisamment longs : « Ilya Selhida » n'est pas protégé si « Selhida » reste.

    `mots_du_texte` contient les mots (en minuscules) présents dans le texte
    original hors données personnelles : un fragment courant comme « des »
    (dans « Route des Alpes ») n'est pas une fuite s'il figure aussi ailleurs.
    """
    sortie = retirer_etiquettes(sortie)
    if valeur_presente(valeur, sortie):
        return False
    fragments = [
        f for f in valeur.split()
        if len(f) >= LONGUEUR_MIN_FRAGMENT and f.lower() not in mots_du_texte
    ]
    return not any(valeur_presente(f, sortie) for f in fragments)


def mots_hors_donnees_personnelles(exemple: dict) -> list[str]:
    """Mots du texte original situés en dehors des données personnelles."""
    texte = exemple["source_text"]
    # On retire les entités en partant de la fin pour ne pas décaler les positions.
    for entite in sorted(exemple["privacy_mask"], key=lambda e: e["start"], reverse=True):
        texte = texte[: entite["start"]] + " " + texte[entite["end"] :]
    return re.findall(r"\w+", texte.lower())


def evaluer_exemple(exemple: dict, sortie: str) -> dict:
    """Évalue la sortie du modèle pour un exemple."""
    mots_attendus = Counter(mots_hors_donnees_personnelles(exemple))
    mots_du_texte = frozenset(mots_attendus)

    par_type = defaultdict(lambda: {"total": 0, "protegees": 0})
    for entite in exemple["privacy_mask"]:
        par_type[entite["label"]]["total"] += 1
        par_type[entite["label"]]["protegees"] += entite_protegee(
            entite["value"], sortie, mots_du_texte
        )

    mots_sortie = Counter(re.findall(r"\w+", retirer_etiquettes(sortie).lower()))
    mots_conserves = sum((mots_attendus & mots_sortie).values())
    ajoutes = mots_ajoutes(exemple["source_text"], sortie)
    inventees = etiquettes_inventees(sortie)

    return {
        "nb_entites": len(exemple["privacy_mask"]),
        "nb_protegees": sum(t["protegees"] for t in par_type.values()),
        "par_type": dict(par_type),
        "nb_mots_attendus": sum(mots_attendus.values()),
        "nb_mots_conserves": mots_conserves,
        "mots_ajoutes": ajoutes,
        "etiquettes_inventees": inventees,
        "fidele": not ajoutes and not inventees,
        "exact": " ".join(sortie.split()) == " ".join(exemple["masked_text"].split()),
    }


def _taux(numerateur: int, denominateur: int) -> float:
    return numerateur / denominateur if denominateur else 1.0


def evaluer(exemples: list[dict], sorties: list[str]) -> dict:
    """Agrège les mesures sur un jeu d'exemples."""
    if len(exemples) != len(sorties):
        raise ValueError(f"{len(exemples)} exemples mais {len(sorties)} sorties.")

    resultats = [evaluer_exemple(ex, s) for ex, s in zip(exemples, sorties)]

    par_type = defaultdict(lambda: {"total": 0, "protegees": 0})
    for r in resultats:
        for label, compte in r["par_type"].items():
            par_type[label]["total"] += compte["total"]
            par_type[label]["protegees"] += compte["protegees"]

    return {
        "protection": _taux(
            sum(r["nb_protegees"] for r in resultats),
            sum(r["nb_entites"] for r in resultats),
        ),
        "textes_sans_fuite": _taux(
            sum(r["nb_protegees"] == r["nb_entites"] for r in resultats), len(resultats)
        ),
        "preservation": _taux(
            sum(r["nb_mots_conserves"] for r in resultats),
            sum(r["nb_mots_attendus"] for r in resultats),
        ),
        "textes_fideles": _taux(sum(r["fidele"] for r in resultats), len(resultats)),
        "correspondance_exacte": _taux(sum(r["exact"] for r in resultats), len(resultats)),
        "protection_par_type": {
            label: _taux(c["protegees"], c["total"])
            for label, c in sorted(par_type.items(), key=lambda kv: -kv[1]["total"])
        },
        # Nombre d'entités par type : un taux sur 3 cas n'est pas significatif.
        "effectifs_par_type": {
            label: c["total"]
            for label, c in sorted(par_type.items(), key=lambda kv: -kv[1]["total"])
        },
    }
