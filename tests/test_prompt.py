from src.prompt import ETIQUETTES, INSTRUCTION, construire_messages


def test_toutes_les_etiquettes_sont_dans_l_instruction():
    for etiquette in ETIQUETTES:
        assert f"[{etiquette}]" in INSTRUCTION


def test_messages_system_puis_user():
    messages = construire_messages("Bonjour Julie.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == INSTRUCTION
    assert messages[1]["content"] == "Bonjour Julie."
