"""Instruction donnée au modèle, identique en évaluation et en entraînement."""

# Types de données personnelles du jeu de données, du plus au moins fréquent.
ETIQUETTES = [
    "GIVENNAME", "SURNAME", "TELEPHONENUM", "CITY", "TIME", "EMAIL", "DATE",
    "STREET", "BUILDINGNUM", "IDCARDNUM", "TITLE", "AGE", "ZIPCODE",
    "PASSPORTNUM", "SEX", "TAXNUM", "SOCIALNUM", "CREDITCARDNUMBER",
    "DRIVERLICENSENUM", "GENDER",
]

INSTRUCTION = (
    "Tu es un outil d'anonymisation de textes. Réécris le texte fourni en remplaçant "
    "chaque donnée personnelle par son type entre crochets, choisi parmi : "
    + ", ".join(f"[{e}]" for e in ETIQUETTES)
    + ". Ne modifie rien d'autre dans le texte. "
    "Réponds uniquement avec le texte anonymisé."
)


def construire_messages(texte: str) -> list[dict]:
    """Conversation à passer au chat template du modèle."""
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": texte},
    ]
