# Anonymisation RGPD de textes français par un petit LLM fine-tuné

Fine-tuning d'un petit modèle de langage (Qwen2.5-0.5B-Instruct) pour détecter et masquer
les données personnelles (noms, téléphones, adresses, e-mails, IBAN...) dans des textes en français.

**Problématique** : les données personnelles ne peuvent pas être envoyées à une API externe
pour être anonymisées. Peut-on obtenir une anonymisation fiable avec un petit modèle
qui tourne en local ?

**Démarche** : préparation des données → SFT (LoRA) → DPO → évaluation et analyse des erreurs.

**Stack** : PyTorch, Hugging Face (transformers, datasets, peft, trl).

> 🚧 Projet en cours.
