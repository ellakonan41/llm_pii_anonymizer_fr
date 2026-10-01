# Anonymisation RGPD de textes français par un petit LLM fine-tuné

Fine-tuning d'un petit modèle de langage, **Qwen2.5-0.5B-Instruct**, pour repérer et masquer
les données personnelles (noms, téléphones, adresses, e-mails, numéros d'identité...) dans des textes en français.

```
Entrée : Bonjour, je suis Sherap Iblikci et j'aimerais commander un puzzle.
Sortie : Bonjour, je suis [GIVENNAME] [SURNAME] et j'aimerais commander un puzzle.
```

> 🚧 Projet en cours : SFT et évaluation sur textes réalistes terminés, DPO à venir.

## Problématique

Pour respecter le RGPD, entreprises, hôpitaux et administrations doivent anonymiser leurs documents
avant de les partager ou de les analyser. Confier cette tâche à une API externe pose un problème
évident : il faudrait lui envoyer les données personnelles que l'on cherche justement à protéger.

**Un petit modèle (0,5 milliard de paramètres), fine-tuné et exécuté en local, peut-il anonymiser
des textes de façon fiable ?**

## Résultats

Deux jeux de test, identiques pour toutes les versions :
- **synthétique** : 500 textes issus du même générateur que les données d'entraînement ;
- **réaliste** : 50 textes français (e-mails, SMS, formulaires médicaux, RH, administratifs...)
  rédigés et vérifiés pour ce projet, avec des noms, adresses et formats français absents des
  données d'entraînement ([guide d'annotation](docs/guide_annotation.md)).

| Métrique | v0, synthétique | **v1, synthétique** | v0, réaliste | **v1, réaliste** |
|---|---|---|---|---|
| Protection (données personnelles masquées) | 42,9 % | **99,7 %** | 25,0 % | **97,0 %** |
| Textes sans aucune fuite | 39,8 % | **99,4 %** | 24,0 % | **88,0 %** |
| Préservation du texte non personnel | 27,2 % | **99,7 %** | 38,3 % | **99,5 %** |
| Correspondance exacte avec la référence | 0,0 % | **77,8 %** | 4,0 % | **48,0 %** |

v0 : modèle de base (zero-shot). v1 : après SFT avec LoRA.

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

Sur les textes réalistes, la v1 reconnaît tous les noms, y compris les cas ambigus (Rose, Marine,
Petit, Blanc, Jean-Baptiste Le Goff, noms en majuscules) et tous les identifiants français
(sécurité sociale, numéro fiscal, permis). Le passage de 99,4 % à 88 % de textes sans fuite
mesure l'écart de généralisation : les fuites se concentrent sur des formats absents des données
synthétiques (voir l'[analyse des erreurs](#7-évaluation-sur-textes-réalistes-v1)).

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

### 7. Évaluation sur textes réalistes (v1)

[Notebook 05](notebooks/05-test-realiste.ipynb). Le jeu de test réaliste est construit avec
[`src/annotation.py`](src/annotation.py) : on écrit un texte et la liste de ses données
personnelles, les positions et la réponse de référence sont calculées automatiquement. Il couvre
les contextes médical, RH, service client, administratif, bancaire et messagerie, avec des cas
difficiles (prénoms qui sont aussi des noms communs, noms composés, formats de téléphone variés,
noms répétés) et 6 textes sans donnée personnelle.

Sur 236 entités, 7 ne sont pas masquées, réparties dans 6 textes :
- **heures au format français** (« 17h15 », « 9h00 ») et **date précédée du jour** (« mardi 8 avril ») :
  les données d'entraînement utilisent surtout d'autres formats (« 11:16 PM ») ;
- **noms de rue composés de mots courants** (« rue de la République », « place de la Mairie »),
  laissés intacts alors que les noms de rue tirés de noms de personnes sont masqués ;
- **deux différences de convention** : le modèle conserve « rue » et masque le nom de la voie
  (« rue [STREET] »), comme dans les données d'entraînement, alors que le guide d'annotation inclut
  le type de voie. La métrique stricte compte ces cas comme des fuites ; sans eux, 92 % des textes
  seraient sans fuite.

Des erreurs de fidélité apparaissent aussi : deux dates fusionnées en une seule étiquette, un nom
d'établissement masqué à tort (« hôpital Pellegrin »), et une faute de frappe introduite dans un
texte sans aucune donnée personnelle (« vestiaires » → « vestiares »).

Ces faiblesses orientent la suite du projet. Les paires de préférences du DPO seront construites
sur des textes d'entraînement distincts du jeu de test, pour que celui-ci reste une mesure fiable.

## Limites

- **Données synthétiques** : phrases parfois artificielles, noms très internationaux, formats
  français sous-représentés ; le jeu de test réaliste ne compte que 50 textes.
- **Fidélité de la recopie** : un petit modèle peut modifier un mot du texte non personnel ;
  une relecture reste nécessaire avant diffusion d'un document anonymisé.
- **Définition des données personnelles** héritée du jeu de données : les heures (TIME) et les
  titres (TITLE) sont considérés comme personnels, ce qui est discutable.
- **Types rares** : quelques cas seulement dans le test (4 pour GENDER, 7 pour SOCIALNUM),
  leurs scores ne sont pas statistiquement significatifs.
- **Métriques** : la préservation ignore la ponctuation ; un fragment de donnée qui figure aussi
  dans le texte non personnel n'est pas compté comme fuite.

## Suite du projet

- [x] Jeu de test de textes français réalistes
- [ ] DPO ciblé sur les faiblesses observées (formats français de dates et d'heures, noms de rue, fidélité)
- [ ] v2 : nettoyage complet (déduplication, plafonnement par structure, échantillonnage des types rares, filtre qualité)
- [ ] Publication de l'adaptateur LoRA sur le Hugging Face Hub

## Structure du dépôt

```
├── notebooks/          # Exploration, préparation, baseline, SFT, test réaliste (exécutés sur Kaggle)
├── src/
│   ├── cleaning.py     # Nettoyage et échantillonnage des données
│   ├── prompt.py       # Consigne et format des exemples, communs à l'entraînement et à l'évaluation
│   ├── generation.py   # Génération des anonymisations par lots
│   ├── evaluation.py   # Métriques : protection, préservation, correspondance exacte
│   └── annotation.py   # Construction d'exemples annotés (jeu de test réaliste)
├── data/
│   └── test_realiste.jsonl  # 50 textes réalistes annotés
├── tests/              # Tests unitaires (pytest)
└── docs/               # Figures et guide d'annotation
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
