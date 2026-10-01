"""Construction d'exemples annotés à la main.

On écrit un texte et la liste des données personnelles qu'il contient ;
les positions et le texte anonymisé de référence sont calculés automatiquement,
au même format que le jeu de données ai4privacy.
"""

import json
import re

from src.prompt import ETIQUETTES


def construire_exemple(texte: str, entites: list[list[str]], uid: str = "") -> dict:
    """Construit un exemple complet à partir d'un texte et de ses entités.

    `entites` est une liste de paires [étiquette, valeur]. Chaque valeur est
    masquée à toutes ses occurrences dans le texte (en mot entier).
    """
    privacy_mask = []
    for label, valeur in entites:
        if label not in ETIQUETTES:
            raise ValueError(f"Étiquette inconnue : {label!r}")
        motif = rf"(?<!\w){re.escape(valeur)}(?!\w)"
        occurrences = list(re.finditer(motif, texte))
        if not occurrences:
            raise ValueError(f"Valeur introuvable dans le texte : {valeur!r}")
        for m in occurrences:
            privacy_mask.append({"label": label, "value": valeur, "start": m.start(), "end": m.end()})

    privacy_mask.sort(key=lambda e: e["start"])
    for precedente, suivante in zip(privacy_mask, privacy_mask[1:]):
        if suivante["start"] < precedente["end"]:
            raise ValueError(f"Entités qui se chevauchent : {precedente['value']!r} et {suivante['value']!r}")

    masque = texte
    for e in reversed(privacy_mask):
        masque = masque[: e["start"]] + f"[{e['label']}]" + masque[e["end"] :]

    return {"uid": uid, "source_text": texte, "masked_text": masque, "privacy_mask": privacy_mask}


def charger_annotations(chemin: str) -> list[dict]:
    """Charge un fichier JSONL d'annotations ({"texte": ..., "entites": [...]}) et construit les exemples."""
    exemples = []
    with open(chemin, encoding="utf-8") as f:
        for numero, ligne in enumerate(f, start=1):
            if not ligne.strip():
                continue
            brut = json.loads(ligne)
            try:
                exemples.append(construire_exemple(brut["texte"], brut["entites"], uid=f"reel-{numero:03d}"))
            except ValueError as erreur:
                raise ValueError(f"Ligne {numero} : {erreur}") from erreur
    return exemples
