import pytest

from src.evaluation import (
    entite_protegee,
    evaluer,
    evaluer_exemple,
    mots_hors_donnees_personnelles,
    valeur_presente,
)

# "Appelez Julie Martin au 06 12 34 56 78."
#          ^8          ^20 ^24            ^38
EXEMPLE = {
    "source_text": "Appelez Julie Martin au 06 12 34 56 78.",
    "masked_text": "Appelez [GIVENNAME] [SURNAME] au [TELEPHONENUM].",
    "privacy_mask": [
        {"label": "GIVENNAME", "value": "Julie", "start": 8, "end": 13},
        {"label": "SURNAME", "value": "Martin", "start": 14, "end": 20},
        {"label": "TELEPHONENUM", "value": "06 12 34 56 78", "start": 24, "end": 38},
    ],
}
SORTIE_PARFAITE = "Appelez [GIVENNAME] [SURNAME] au [TELEPHONENUM]."


def test_positions_de_l_exemple_de_test():
    for e in EXEMPLE["privacy_mask"]:
        assert EXEMPLE["source_text"][e["start"] : e["end"]] == e["value"]


# --- valeur_presente ---

def test_trouve_une_valeur_presente():
    assert valeur_presente("Julie", "Appelez Julie demain.")


def test_ignore_la_casse():
    assert valeur_presente("Julie", "appelez JULIE demain.")


def test_ne_confond_pas_avec_un_morceau_de_mot():
    assert not valeur_presente("Ali", "Une analyse de qualité.")


# --- entite_protegee ---

def test_entite_absente_est_protegee():
    assert entite_protegee("Julie", "Appelez [GIVENNAME] demain.")


def test_entite_intacte_n_est_pas_protegee():
    assert not entite_protegee("Julie", "Appelez Julie demain.")


def test_fragment_restant_est_une_fuite():
    assert not entite_protegee("Ilya Selhida", "[GIVENNAME] Selhida a répondu.")


def test_fragment_court_restant_n_est_pas_une_fuite():
    # « 06 » seul est trop court pour être attribué au numéro masqué.
    assert entite_protegee("06 12 34 56 78", "[TELEPHONENUM], rendez-vous le 06.")


def test_fragment_retrouve_dans_une_etiquette_n_est_pas_une_fuite():
    # « Street » ne doit pas être retrouvé dans l'étiquette [STREET].
    assert entite_protegee("Winston Street", "Rendez-vous : [STREET], [CITY].")


def test_fragment_present_ailleurs_dans_le_texte_n_est_pas_une_fuite():
    # « des » figure aussi dans « Liste des rues », hors donnée personnelle.
    assert entite_protegee(
        "Route des Grandes Alpes",
        "Liste des rues : [STREET].",
        mots_du_texte=frozenset({"liste", "des", "rues"}),
    )


def test_fragment_absent_du_reste_du_texte_reste_une_fuite():
    assert not entite_protegee(
        "Route des Grandes Alpes",
        "Liste des rues : Grandes Alpes.",
        mots_du_texte=frozenset({"liste", "des", "rues"}),
    )


# --- mots_hors_donnees_personnelles ---

def test_mots_hors_donnees_personnelles():
    assert mots_hors_donnees_personnelles(EXEMPLE) == ["appelez", "au"]


# --- evaluer_exemple ---

def test_sortie_parfaite():
    r = evaluer_exemple(EXEMPLE, SORTIE_PARFAITE)
    assert r["nb_protegees"] == r["nb_entites"] == 3
    assert r["nb_mots_conserves"] == r["nb_mots_attendus"] == 2
    assert r["exact"]


def test_sortie_qui_oublie_le_nom():
    r = evaluer_exemple(EXEMPLE, "Appelez [GIVENNAME] Martin au [TELEPHONENUM].")
    assert r["nb_protegees"] == 2
    assert r["par_type"]["SURNAME"] == {"total": 1, "protegees": 0}
    assert not r["exact"]


def test_sortie_qui_masque_tout_ne_preserve_rien():
    r = evaluer_exemple(EXEMPLE, "[GIVENNAME]")
    assert r["nb_protegees"] == 3
    assert r["nb_mots_conserves"] == 0


def test_etiquette_differente_mais_donnee_protegee():
    # Nom complet en une seule étiquette : pas de fuite, mais pas identique à la référence.
    r = evaluer_exemple(EXEMPLE, "Appelez [GIVENNAME] au [TELEPHONENUM].")
    assert r["nb_protegees"] == 3
    assert not r["exact"]


def test_reference_avec_mot_courant_dans_une_ville_est_sans_fuite():
    # Cas réel : « Ville » figure aussi dans « ma ville natale ».
    source = "Ma ville natale, Nantes Centre Ville, a changé."
    exemple = {
        "source_text": source,
        "masked_text": "Ma ville natale, [CITY], a changé.",
        "privacy_mask": [{"label": "CITY", "value": "Nantes Centre Ville", "start": 17, "end": 36}],
    }
    assert source[17:36] == "Nantes Centre Ville"
    r = evaluer_exemple(exemple, exemple["masked_text"])
    assert r["nb_protegees"] == 1


def test_etiquettes_ignorees_pour_la_preservation():
    # « street » hors donnée personnelle ne doit pas être « conservé » via l'étiquette [STREET].
    exemple = {
        "source_text": "Street art à Lyon.",
        "masked_text": "Street art à [CITY].",
        "privacy_mask": [{"label": "CITY", "value": "Lyon", "start": 13, "end": 17}],
    }
    r = evaluer_exemple(exemple, "[STREET] art à [CITY].")
    assert r["nb_mots_conserves"] == r["nb_mots_attendus"] - 1


def test_espaces_superflus_ignores_pour_la_correspondance_exacte():
    assert evaluer_exemple(EXEMPLE, "  Appelez [GIVENNAME]  [SURNAME] au [TELEPHONENUM].\n")["exact"]


# --- evaluer ---

def test_evaluer_agrege_les_exemples():
    resultats = evaluer(
        [EXEMPLE, EXEMPLE],
        [SORTIE_PARFAITE, "Appelez [GIVENNAME] Martin au [TELEPHONENUM]."],
    )
    assert resultats["protection"] == pytest.approx(5 / 6)
    assert resultats["textes_sans_fuite"] == 0.5
    assert resultats["preservation"] == 1.0
    assert resultats["correspondance_exacte"] == 0.5
    assert resultats["protection_par_type"]["SURNAME"] == 0.5
    assert resultats["protection_par_type"]["GIVENNAME"] == 1.0


def test_texte_sans_entite_compte_comme_sans_fuite():
    exemple = {"source_text": "Merci pour tout.", "masked_text": "Merci pour tout.", "privacy_mask": []}
    resultats = evaluer([exemple], ["Merci pour tout."])
    assert resultats["textes_sans_fuite"] == 1.0
    assert resultats["preservation"] == 1.0


def test_evaluer_refuse_des_listes_de_tailles_differentes():
    with pytest.raises(ValueError):
        evaluer([EXEMPLE], [])
