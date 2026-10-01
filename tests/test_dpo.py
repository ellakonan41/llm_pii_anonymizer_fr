import random
from pathlib import Path

from src.annotation import charger_annotations, construire_exemple
from src.dpo import (
    NOMS,
    PRENOMS,
    RUES,
    VILLES,
    anonymiser_sauf,
    choisir_rejet,
    corrompre,
    generer_exemples,
    introduire_faute,
    masquer_en_trop,
    paire_dpo,
)
from src.evaluation import evaluer_exemple
from src.prompt import construire_messages

TEST_REALISTE = Path(__file__).parent.parent / "data" / "test_realiste.jsonl"

EXEMPLE = construire_exemple(
    "Rendez-vous mardi 8 avril à 17h15 au 4 rue de la Liberté.",
    [["DATE", "8 avril"], ["TIME", "17h15"], ["BUILDINGNUM", "4"], ["STREET", "rue de la Liberté"]],
)


# --- Séparation avec le jeu de test réaliste ---

def test_aucune_valeur_du_test_realiste_dans_les_listes():
    valeurs_test = {e["value"] for ex in charger_annotations(str(TEST_REALISTE)) for e in ex["privacy_mask"]}
    communes = valeurs_test & set(PRENOMS + NOMS + VILLES + RUES)
    assert not communes, f"Valeurs partagées avec le test réaliste : {communes}"


def test_aucun_texte_genere_identique_au_test_realiste():
    textes_test = {ex["source_text"] for ex in charger_annotations(str(TEST_REALISTE))}
    assert not textes_test & {ex["source_text"] for ex in generer_exemples(500)}


# --- Génération ---

def test_generation_reproductible():
    assert generer_exemples(20, graine=1) == generer_exemples(20, graine=1)


def test_exemples_generes_valides_et_coherents():
    for ex in generer_exemples(300):
        r = evaluer_exemple(ex, ex["masked_text"])
        assert r["nb_protegees"] == r["nb_entites"]
        assert r["nb_mots_conserves"] == r["nb_mots_attendus"]


def test_generation_contient_des_textes_sans_donnee():
    exemples = generer_exemples(300)
    assert any(not ex["privacy_mask"] for ex in exemples)


# --- Corruptions ---

def test_anonymiser_sauf_laisse_une_entite_en_clair():
    assert anonymiser_sauf(EXEMPLE, {1}) == "Rendez-vous mardi [DATE] à 17h15 au [BUILDINGNUM] [STREET]."


def test_anonymiser_sauf_sans_exception_donne_la_reference():
    assert anonymiser_sauf(EXEMPLE, set()) == EXEMPLE["masked_text"]


def test_introduire_faute_modifie_un_seul_mot_sans_toucher_aux_etiquettes():
    texte = "Rendez-vous [DATE] au cabinet médical."
    fautif = introduire_faute(texte, random.Random(0))
    assert len(fautif) == len(texte) - 1
    assert "[DATE]" in fautif


def test_masquer_en_trop_ajoute_une_etiquette():
    texte = "Rendez-vous [DATE] au cabinet médical."
    rejet = masquer_en_trop(texte, random.Random(0))
    assert rejet != texte
    assert rejet.count("[") == 2


def test_corrompre_produit_une_reponse_differente_et_moins_bonne():
    rng = random.Random(0)
    for _ in range(30):
        rejet, strategie = corrompre(EXEMPLE, rng)
        assert rejet != EXEMPLE["masked_text"]
        r = evaluer_exemple(EXEMPLE, rejet)
        if strategie == "fuite":
            assert r["nb_protegees"] < r["nb_entites"]


def test_corrompre_un_texte_sans_donnee():
    ex = construire_exemple("Merci de fermer les fenêtres en partant.", [])
    rejet, strategie = corrompre(ex, random.Random(0))
    assert strategie in {"faute", "en_trop"}
    assert rejet != ex["masked_text"]


# --- Choix de la réponse rejetée ---

def test_erreur_du_modele_utilisee_comme_rejet():
    sortie = "Rendez-vous mardi [DATE] à 17h15 au [BUILDINGNUM] [STREET]."
    assert choisir_rejet(EXEMPLE, sortie, random.Random(0)) == (sortie, "modele")


def test_sortie_correcte_du_modele_remplacee_par_une_erreur_construite():
    rejet, strategie = choisir_rejet(EXEMPLE, EXEMPLE["masked_text"], random.Random(0))
    assert strategie != "modele"
    assert rejet != EXEMPLE["masked_text"]


def test_paire_dpo_au_format_conversationnel():
    paire = paire_dpo(EXEMPLE, "rejet")
    assert paire["prompt"] == construire_messages(EXEMPLE["source_text"])
    assert paire["chosen"] == [{"role": "assistant", "content": EXEMPLE["masked_text"]}]
    assert paire["rejected"] == [{"role": "assistant", "content": "rejet"}]
