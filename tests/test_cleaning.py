import pytest

from src.cleaning import (
    est_valide,
    normaliser_etiquettes,
    retirer_fuite,
    tirer_echantillon,
)


def exemple(source, masque="[GIVENNAME]"):
    return {"source_text": source, "masked_text": masque}


# --- normaliser_etiquettes ---

def test_retire_le_numero_d_une_etiquette():
    assert normaliser_etiquettes("Bonjour [GIVENNAME_1].") == "Bonjour [GIVENNAME]."


def test_retire_les_numeros_de_plusieurs_etiquettes():
    texte = "Le [AGE_2] minimum est de [AGE_1] ans."
    assert normaliser_etiquettes(texte) == "Le [AGE] minimum est de [AGE] ans."


def test_laisse_intact_un_texte_sans_etiquette_numerotee():
    texte = "Merci pour votre participation [sic]."
    assert normaliser_etiquettes(texte) == texte


# --- est_valide ---

def test_accepte_un_exemple_normal():
    assert est_valide(exemple("Bonjour Oswald.", "Bonjour [GIVENNAME_1]."))


@pytest.mark.parametrize("source", ["", "   ", "..."])
def test_rejette_un_texte_original_sans_contenu(source):
    assert not est_valide(exemple(source))


def test_rejette_un_texte_masque_reduit_a_des_points():
    assert not est_valide(exemple("Bonjour Oswald.", "..."))


# --- retirer_fuite ---

def test_retire_du_train_les_textes_presents_en_validation():
    train = [exemple("A"), exemple("B"), exemple("C")]
    validation = [exemple("B")]
    assert retirer_fuite(train, validation) == [exemple("A"), exemple("C")]


def test_ne_retire_rien_sans_texte_commun():
    train = [exemple("A"), exemple("B")]
    assert retirer_fuite(train, [exemple("Z")]) == train


# --- tirer_echantillon ---

def test_echantillon_de_la_bonne_taille():
    exemples = [exemple(str(i)) for i in range(100)]
    assert len(tirer_echantillon(exemples, 10)) == 10


def test_meme_graine_meme_echantillon():
    exemples = [exemple(str(i)) for i in range(100)]
    assert tirer_echantillon(exemples, 10, graine=1) == tirer_echantillon(exemples, 10, graine=1)


def test_graines_differentes_echantillons_differents():
    exemples = [exemple(str(i)) for i in range(100)]
    assert tirer_echantillon(exemples, 10, graine=1) != tirer_echantillon(exemples, 10, graine=2)


def test_echantillon_trop_grand_leve_une_erreur():
    with pytest.raises(ValueError):
        tirer_echantillon([exemple("A")], 5)
