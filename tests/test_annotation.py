import json

import pytest

from src.annotation import charger_annotations, construire_exemple


def test_calcule_positions_et_texte_masque():
    ex = construire_exemple(
        "Appelez Marie Dupont au 06 12 34 56 78.",
        [["GIVENNAME", "Marie"], ["SURNAME", "Dupont"], ["TELEPHONENUM", "06 12 34 56 78"]],
    )
    assert ex["masked_text"] == "Appelez [GIVENNAME] [SURNAME] au [TELEPHONENUM]."
    for e in ex["privacy_mask"]:
        assert ex["source_text"][e["start"] : e["end"]] == e["value"]


def test_masque_toutes_les_occurrences():
    ex = construire_exemple("Marie a appelé. Rappelez Marie.", [["GIVENNAME", "Marie"]])
    assert ex["masked_text"] == "[GIVENNAME] a appelé. Rappelez [GIVENNAME]."
    assert len(ex["privacy_mask"]) == 2


def test_ne_masque_pas_un_morceau_de_mot():
    ex = construire_exemple("Pierre aime la pierre de Pierrefonds.", [["GIVENNAME", "Pierre"]])
    assert ex["masked_text"] == "[GIVENNAME] aime la pierre de Pierrefonds."


def test_texte_sans_entite():
    ex = construire_exemple("Merci pour votre retour.", [])
    assert ex["masked_text"] == "Merci pour votre retour."
    assert ex["privacy_mask"] == []


def test_etiquette_inconnue():
    with pytest.raises(ValueError, match="Étiquette inconnue"):
        construire_exemple("Bonjour Marie.", [["PRENOM", "Marie"]])


def test_valeur_absente_du_texte():
    with pytest.raises(ValueError, match="introuvable"):
        construire_exemple("Bonjour Marie.", [["GIVENNAME", "Julie"]])


def test_entites_qui_se_chevauchent():
    with pytest.raises(ValueError, match="chevauchent"):
        construire_exemple("Rue Victor Hugo", [["STREET", "Rue Victor Hugo"], ["SURNAME", "Hugo"]])


def test_charger_annotations(tmp_path):
    chemin = tmp_path / "annotations.jsonl"
    lignes = [
        {"texte": "Bonjour Marie.", "entites": [["GIVENNAME", "Marie"]]},
        {"texte": "Merci.", "entites": []},
    ]
    chemin.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lignes), encoding="utf-8")
    exemples = charger_annotations(str(chemin))
    assert [ex["uid"] for ex in exemples] == ["reel-001", "reel-002"]
    assert exemples[0]["masked_text"] == "Bonjour [GIVENNAME]."


def test_charger_annotations_indique_la_ligne_en_erreur(tmp_path):
    chemin = tmp_path / "annotations.jsonl"
    chemin.write_text('{"texte": "Bonjour.", "entites": [["GIVENNAME", "Marie"]]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Ligne 1"):
        charger_annotations(str(chemin))
