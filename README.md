# Anonymisation RGPD de textes français par un petit LLM fine-tuné

Fine-tuning d'un petit modèle de langage, **Qwen2.5-0.5B-Instruct**, pour repérer et masquer
les données personnelles (noms, téléphones, adresses, e-mails, numéros d'identité...) dans des textes en français.

```
Entrée : Bonjour, je suis Sherap Iblikci et j'aimerais commander un puzzle.
Sortie : Bonjour, je suis [GIVENNAME] [SURNAME] et j'aimerais commander un puzzle.
```

> 🚧 Projet en cours : SFT terminé, test sur textes réalistes et DPO à venir.

## Problématique

Pour respecter le RGPD, entreprises, hôpitaux et administrations doivent anonymiser leurs documents
avant de les partager ou de les analyser. Confier cette tâche à une API externe pose un problème
évident : il faudrait lui envoyer les données personnelles que l'on cherche justement à protéger.

**Un petit modèle (0,5 milliard de paramètres), fine-tuné et exécuté en local, peut-il anonymiser
des textes de façon fiable ?**

## Résultats

Évaluation sur 500 textes de test, identiques pour toutes les versions.

| Métrique | v0 : modèle de base (zero-shot) | **v1 : après SFT (LoRA)** |
|---|---|---|
| Protection (données personnelles masquées) | 42,9 % | **99,7 %** |
| Textes sans aucune fuite | 39,8 % | **99,4 %** |
| Préservation du texte non personnel | 27,2 % | **99,7 %** |
| Correspondance exacte avec la référence | 0,0 % | **77,8 %** |

Le modèle de base ne suit pas la consigne : il recopie le texte sans le masquer, le reformule en
listes, répond aux questions qu'il contient ou modifie des noms. Seules 14,8 % de ses sorties
contiennent une étiquette, et 19,6 % font moins de la moitié du texte d'origine : sa protection
apparente vient surtout de la destruction du texte, comme le montre sa faible préservation.
Après fine-tuning sur 4 800 exemples, en n'entraînant que 1,75 % des paramètres, le modèle masque
correctement les données et conserve le reste du texte.

| Texte | v0 | v1 |
|---|---|---|
| Bonjour, je suis Sherap Iblikci et j'aimerais commander un puzzle… | `**Sherap Iblikci**` `**Commande**` `**Puzzle**`… | Bonjour, je suis [GIVENNAME] [SURNAME] et j'aimerais commander un puzzle… |
| Disposez-vous d'une assurance responsabilité civile ? | `[YES]` | Disposez-vous d'une assurance responsabilité civile ? |

> ⚠️ Le jeu de test provient du même générateur synthétique que les données d'entraînement.
> Ces scores mesurent la maîtrise de la tâche sur cette distribution, pas encore la performance
> sur des documents réels : un jeu de test de textes français réalistes est en préparation.

## Démarche

### 1. Données

Jeu de données [`ai4privacy/open-pii-masking-500k-ai4privacy`](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy)
(licence CC-BY-4.0), filtré sur le français : 89 670 exemples d'entraînement et 22 466 de validation.
Chaque exemple associe un texte, sa version anonymisée et la position de chaque donnée personnelle.

Le choix de ce jeu de données a été guidé par sa licence : la version `pii-masking-300k`, plus connue,
interdit la diffusion d'œuvres dérivées, et donc la publication d'un modèle entraîné dessus.

L'exploration ([notebook 01](notebooks/01-exploration-donnees.ipynb)) a mis en évidence :
- des **artefacts de génération** : valeurs insérées à la place de noms communs (« Le 30 minimum
  pour participer… »), codes postaux non français, numérotation incohérente des étiquettes ;
- une **annotation incohérente** des noms complets, tantôt en une étiquette, tantôt en deux ;
- une **distribution déséquilibrée** : prénoms et noms représentent ~44 % des entités, les
  identifiants sensibles (carte bancaire, sécurité sociale) moins de 1 % chacun ;
- des textes courts (médiane de 120 caractères) et peu de doublons : 180 doublons exacts,
  51 textes communs à l'entraînement et à la validation, et seulement 2,6 % de phrases
  construites sur une même structure.

### 2. Nettoyage minimal et jeux de données

Pour la v1, un nettoyage volontairement minimal ([`src/cleaning.py`](src/cleaning.py)), afin de
mesurer plus tard l'apport d'un nettoyage complet :
- retrait des textes sans contenu (« ... », « - ») ;
- retrait de l'entraînement des textes également présents en validation (fuite) ;
- normalisation des étiquettes (`[GIVENNAME_1]` → `[GIVENNAME]`).

Jeux obtenus ([notebook 02](notebooks/02-preparation-donnees.ipynb)), par tirage reproductible :
**500 textes de test**, issus de la validation et figés pour toutes les versions, et
**5 000 textes d'entraînement**, dont 200 réservés au suivi de la loss de validation.

### 3. Métriques

Une anonymisation se juge sur deux axes opposés, comme le rappel et la précision
([`src/evaluation.py`](src/evaluation.py)) :
- **protection** : part des données personnelles qui ont disparu de la sortie, fragments compris
  (« Ilya Selhida » n'est pas protégé si « Selhida » reste) ; un oubli est une fuite ;
- **préservation** : part des mots non personnels conservés, sans quoi un modèle qui efface
  tout obtiendrait une protection parfaite ;
- **textes sans fuite** : le point de vue de l'utilisateur, un seul oubli suffisant à rendre
  un document non conforme ;
- **correspondance exacte** : indicative, pénalisée par les incohérences d'annotation.

La protection ne vérifie pas le type d'étiquette choisi, seulement la disparition de la donnée :
c'est ce qui compte pour le RGPD, et le jeu de données n'est pas cohérent sur ce point.

**Contrôle de cohérence** : appliquée aux réponses de référence, la métrique devait donner 100 %.
Elle donnait 99,5 % : six faux positifs, causés par des fragments retrouvés dans les étiquettes
elles-mêmes (« Street » dans `[STREET]`) ou par des mots courants présents ailleurs dans le texte
(« ville », « des »). Corrigée et couverte par des tests avant toute évaluation de modèle.

### 4. Baseline (v0)

Le modèle de base, évalué sans entraînement avec la même consigne
([notebook 03](notebooks/03-baseline-v0.ipynb)), en décodage glouton (reproductible).

### 5. SFT avec LoRA (v1)

[Notebook 04](notebooks/04-sft-v1.ipynb), avec `trl` (`SFTTrainer`) et `peft` :
- exemples au format conversationnel *prompt / completion*, avec la même consigne qu'à
  l'évaluation ([`src/prompt.py`](src/prompt.py)) ; la loss n'est calculée que sur la réponse,
  ce qui a été vérifié sur un batch réel ;
- LoRA de rang 16 sur l'attention et le MLP : 8,8 M paramètres entraînés sur 494 M (1,75 %) ;
- 1 époque, batch effectif de 16 (300 steps), learning rate 2e-4 avec warmup et décroissance
  cosinus, précision mixte fp16 ; environ 10 minutes sur un GPU T4 (Kaggle).

![Évolution de la loss pendant le SFT](docs/loss_v1.png)

La loss chute en une vingtaine de steps, le temps d'apprendre le format de réponse, puis décroît
lentement. La loss de validation ne remonte jamais : pas de surapprentissage. Une loss très basse
ne suffit pas à juger le modèle, car elle est dominée par les tokens simplement recopiés : seules
les métriques de la tâche disent si les données personnelles sont bien masquées.

### 6. Analyse des erreurs (v1)

Sur les 3 textes signalés comme fuites :
- **une vraie fuite partielle** : un prénom rare et ambigu (« Wiki Emilce » → « Wiki [GIVENNAME] ») ;
- **une erreur d'annotation** du jeu de données : « les années 27 », étiqueté comme un âge, que
  le modèle a eu raison de ne pas masquer ;
- **un cas discutable** : un titre (« Mademoiselle ») non masqué, alors que le nom qui suit l'est.

Les écarts à la référence sans fuite viennent surtout d'ambiguïtés héritées des données : noms
complets en une ou deux étiquettes, confusion entre code postal et numéro de rue.

## Limites

- **Données synthétiques** : phrases parfois artificielles, noms très internationaux et peu
  représentatifs des noms français courants ; le test actuel est de même distribution que l'entraînement.
- **Définition des données personnelles** héritée du jeu de données : les heures (TIME) et les
  titres (TITLE) sont considérés comme personnels, ce qui est discutable.
- **Types rares** : quelques cas seulement dans le test (4 pour GENDER, 7 pour SOCIALNUM),
  leurs scores ne sont pas statistiquement significatifs.
- **Métriques** : la préservation ignore la ponctuation ; un fragment de donnée qui figure aussi
  dans le texte non personnel n'est pas compté comme fuite.

## Suite du projet

- [ ] Jeu de test de textes français réalistes, écrits et annotés à la main
- [ ] v2 : nettoyage complet (déduplication, plafonnement par structure, échantillonnage des types rares, filtre qualité)
- [ ] v3 : DPO à partir des erreurs du modèle
- [ ] Publication de l'adaptateur LoRA sur le Hugging Face Hub

## Structure du dépôt

```
├── notebooks/          # Exploration, préparation, baseline, SFT (exécutés sur Kaggle)
├── src/
│   ├── cleaning.py     # Nettoyage et échantillonnage des données
│   ├── prompt.py       # Consigne et format des exemples, communs à l'entraînement et à l'évaluation
│   ├── generation.py   # Génération des anonymisations par lots
│   └── evaluation.py   # Métriques : protection, préservation, correspondance exacte
├── tests/              # Tests unitaires (pytest)
└── docs/               # Figures
```

## Reproduire

Les notebooks s'exécutent sur Kaggle (GPU T4) et importent le code de `src/` en clonant ce dépôt.
Versions utilisées : `trl 0.29.1`, `peft 0.19.1`, `transformers 5.0.0`.
Sur Kaggle, désinstaller `torchao` (version préinstallée incompatible avec `peft`).

Tests en local :

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest
```

## Données

Ce projet utilise le jeu de données *Open PII Masking 500k* d'Ai4Privacy
([DOI 10.57967/hf/4852](https://doi.org/10.57967/hf/4852)), sous licence CC-BY-4.0.
