"""Génération des anonymisations par un modèle, identique pour toutes les versions évaluées."""

import torch
from tqdm.auto import tqdm

from src.prompt import construire_messages


@torch.no_grad()
def anonymiser(model, tokenizer, textes: list[str], taille_lot: int = 16, max_new_tokens: int = 512) -> list[str]:
    """Anonymise une liste de textes par lots, en décodage glouton (reproductible).

    Le tokenizer doit compléter à gauche (`padding_side = "left"`) : le modèle
    génère à droite du prompt.
    """
    if tokenizer.padding_side != "left":
        raise ValueError("Le tokenizer doit avoir padding_side = 'left' pour la génération par lots.")

    sorties = []
    for i in tqdm(range(0, len(textes), taille_lot)):
        lot = textes[i : i + taille_lot]
        prompts = [
            tokenizer.apply_chat_template(construire_messages(t), tokenize=False, add_generation_prompt=True)
            for t in lot
        ]
        entrees = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        generes = model.generate(**entrees, max_new_tokens=max_new_tokens, do_sample=False)
        nouveaux = generes[:, entrees["input_ids"].shape[1] :]
        sorties += [s.strip() for s in tokenizer.batch_decode(nouveaux, skip_special_tokens=True)]
    return sorties
