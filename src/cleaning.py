"""Nettoyage des exemples du jeu de données ai4privacy.

Un exemple est un dictionnaire contenant au moins les clés
`source_text` (texte original) et `masked_text` (texte anonymisé).
"""

import random
import re

# [GIVENNAME_1] -> [GIVENNAME]
_ETIQUETTE_NUMEROTEE = re.compile(r"\[([A-Z]+)_\d+\]")

# Au moins une lettre ou un chiffre
_CONTENU = re.compile(r"\w")


def normaliser_etiquettes(texte: str) -> str:
    """Retire la numérotation des étiquettes : [GIVENNAME_1] devient [GIVENNAME]."""
    return _ETIQUETTE_NUMEROTEE.sub(r"[\1]", texte)


def est_valide(exemple: dict) -> bool:
    """Indique si un exemple est exploitable.

    Rejette les textes vides ou sans aucun contenu (ex. « ... »),
    côté texte original comme côté texte anonymisé.
    """
    return bool(
        _CONTENU.search(exemple["source_text"])
        and _CONTENU.search(exemple["masked_text"])
    )


def retirer_fuite(train: list[dict], validation: list[dict]) -> list[dict]:
    """Retire du train les exemples dont le texte original apparaît aussi en validation."""
    textes_validation = {ex["source_text"] for ex in validation}
    return [ex for ex in train if ex["source_text"] not in textes_validation]


def tirer_echantillon(exemples: list[dict], n: int, graine: int = 42) -> list[dict]:
    """Tire n exemples au hasard, de façon reproductible (même graine = même tirage)."""
    if n > len(exemples):
        raise ValueError(f"Échantillon de {n} demandé, seulement {len(exemples)} exemples.")
    return random.Random(graine).sample(exemples, n)
