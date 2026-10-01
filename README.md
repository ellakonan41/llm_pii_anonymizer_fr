# Anonymisation RGPD de textes français par un petit LLM fine-tuné

Fine-tuning d'un petit modèle de langage, **Qwen2.5-0.5B-Instruct**, pour repérer et masquer
les données personnelles (noms, téléphones, adresses, e-mails, numéros d'identité...) dans des textes en français.

```
Entrée : Bonjour, je suis Sherap Iblikci et j'aimerais commander un puzzle.
Sortie : Bonjour, je suis [GIVENNAME] [SURNAME] et j'aimerais commander un puzzle.
```

> SFT, évaluation sur textes réalistes et premier essai de DPO réalisés. Un second réglage du
> DPO et un second jeu de test réaliste sont en cours.

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

**Test réaliste (50 textes)**

| Métrique | v0 : modèle de base | **v1 : SFT** | v3 : SFT + DPO |
|---|---|---|---|
| Protection (données personnelles masquées) | 25,0 % | **97,0 %** | 100 % |
| Textes sans aucune fuite | 24,0 % | **88,0 %** | 100 % |
| Préservation du texte non personnel | 38,3 % | **99,5 %** | 98,1 % |
| Correspondance exacte avec la référence | 4,0 % | **48,0 %** | 54,0 % |

**Test synthétique (500 textes)**

| Métrique | v0 : modèle de base | **v1 : SFT** | v3 : SFT + DPO |
|---|---|---|---|
| Protection | 42,9 % | **99,5 %** | 99,9 % |
| Textes sans aucune fuite | 39,8 % | **99,0 %** | 99,8 % |
| Préservation | 27,2 % | **99,7 %** | 99,3 % |
| Correspondance exacte | 0,0 % | **76,8 %** | 68,0 % |

v0 : modèle de base (zero-shot). v1 : SFT avec LoRA. v3 : DPO à partir de la v1.
Le DPO supprime toutes les fuites du test réaliste, mais introduit des masquages injustifiés et
des modifications du texte (voir [l'analyse](#8-dpo-v3)) : **la v1 reste la version de référence**.

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
(sécurité sociale, numéro fiscal, permis). Le passage de 99 % à 88 % de textes sans fuite
mesure l'écart de généralisation : les fuites se concentrent sur des formats absents des données
synthétiques (voir l'[analyse des erreurs](#7-évaluation-sur-textes-réalistes-v1)).

Les chiffres de la v1 sont ceux de l'adaptateur sauvegardé. Un premier entraînement, avec les
mêmes données et réglages, avait obtenu 99,7 % de protection et 99,4 % de textes sans fuite sur le
test synthétique : l'écart vient du non-déterminisme de certains calculs sur GPU.

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
- **textes fidèles** : part des textes où le modèle n'ajoute aucun mot absent du texte original et
  n'invente aucune étiquette ; ajoutée après le DPO, car la préservation ne voit pas ce que le
  modèle ajoute ou modifie (« 5 jours » réécrit en « [AGE] ans ») ;
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

Analyse réalisée sur le premier entraînement. Sur les 3 textes signalés comme fuites :
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

### 8. DPO (v3)

**Données** ([`src/dpo.py`](src/dpo.py)) : 1 200 paires de préférences, à partir de 800 textes
générés par gabarits ciblant les faiblesses observées (heures et dates au format français, noms de
rue composés de mots courants, phrases sans donnée personnelle) et de 400 textes ai4privacy jamais
utilisés (sans adresse, les deux sources n'annotant pas les rues de la même façon). Les noms, villes
et rues des gabarits sont distincts de ceux du test réaliste, ce que vérifie un test automatique.

La réponse rejetée est l'erreur de la v1 elle-même lorsqu'elle se trompe : c'est le cas pour
**335 textes sur 1 200 (28 %)**, ce qui confirme que les gabarits ciblent ses vraies faiblesses.
Sinon, une erreur est construite à partir de la bonne réponse : donnée remise en clair, faute de
frappe ou mot masqué à tort.

**Entraînement** ([notebook 06](notebooks/06-dpo-v3.ipynb)) : `DPOTrainer` de `trl`, nouvel
adaptateur LoRA sur la v1 fusionnée, qui sert aussi de modèle de référence ; `beta` = 0,1, loss DPO
combinée à une loss SFT sur la réponse choisie, learning rate 2e-5, batch effectif de 16.
Un premier essai a dépassé la mémoire du GPU : le DPO calcule les logits de deux réponses par
exemple, pour le modèle et sa référence, sur un vocabulaire de 152 000 tokens. Le batch a été
réduit et compensé par l'accumulation de gradients.

![Loss et récompenses implicites pendant le DPO](docs/dpo_v3.png)

L'entraînement se déroule comme attendu : sur la validation, le modèle préfère la réponse choisie
dans 91 à 93 % des paires, la marge entre réponses choisie et rejetée passe de 0,9 à 2,6, et la
récompense de la réponse choisie reste stable grâce à la loss SFT pendant que celle de la réponse
rejetée chute.

**Résultats** : toutes les fuites du test réaliste disparaissent (100 % de textes sans fuite,
heures, dates et rues comprises). Mais l'analyse des sorties révèle une **sur-optimisation** :
- des **masquages injustifiés** : « carte Vitale » → « carte [CITY] », « lundi de Pâques » →
  « lundi de [DATE] », « salle B », « Formulaire CAF » ;
- des **modifications du texte** : « paracétamol 1 g … pendant 5 jours » devient
  « paracétamol [量] g … pendant [AGE] ans », une étiquette inexistante est inventée
  (`[RESOURCES_HUMANITIES]`), des mots sont supprimés (« Assuré : », « née le »).

Poussé à éviter les fuites, le modèle masque dans le doute et s'éloigne de la v1 au point de perdre
en stabilité. Ces défauts restaient presque invisibles dans les métriques (préservation de 99,5 % à
98,1 %), qui ne détectaient pas les mots ajoutés ou modifiés : d'où l'ajout de la métrique de
fidélité. Pour un outil d'anonymisation, modifier le contenu d'une ordonnance est aussi grave qu'une
fuite : **la v3 n'est pas retenue**.

Deux réserves sur ces résultats : le test réaliste a servi à identifier les faiblesses que le DPO
cible, ses scores sur la v3 sont donc probablement optimistes ; et sur le test synthétique, la baisse
de correspondance exacte s'explique en partie par la convention d'annotation des rues apprise
pendant le DPO, différente de celle du jeu de données.

**Prochaines étapes** : un DPO plus conservateur (`beta` plus élevé, learning rate plus faible,
davantage de paires contre les masquages injustifiés et les modifications du texte), évalué avec la
métrique de fidélité sur un second jeu de test réaliste, jamais utilisé pour orienter l'entraînement.

## Limites

- **Données synthétiques** : phrases parfois artificielles, noms très internationaux, formats
  français sous-représentés ; le jeu de test réaliste ne compte que 50 textes.
- **Fidélité de la recopie** : un petit modèle peut modifier un mot du texte non personnel ;
  une relecture reste nécessaire avant diffusion d'un document anonymisé.
- **Test réaliste réutilisé** : ayant servi à orienter le DPO, il ne constitue plus une mesure
  indépendante pour la v3.
- **Définition des données personnelles** héritée du jeu de données : les heures (TIME) et les
  titres (TITLE) sont considérés comme personnels, ce qui est discutable.
- **Types rares** : quelques cas seulement dans le test (4 pour GENDER, 7 pour SOCIALNUM),
  leurs scores ne sont pas statistiquement significatifs.
- **Métriques** : la préservation ignore la ponctuation ; un fragment de donnée qui figure aussi
  dans le texte non personnel n'est pas compté comme fuite.

## Suite du projet

- [x] Jeu de test de textes français réalistes
- [x] Premier DPO ciblé sur les faiblesses observées
- [x] Métrique de fidélité (mots ajoutés, étiquettes inventées)
- [ ] DPO plus conservateur, évalué sur un second jeu de test réaliste
- [ ] v2 : nettoyage complet (déduplication, plafonnement par structure, échantillonnage des types rares, filtre qualité)
- [ ] Publication de l'adaptateur LoRA sur le Hugging Face Hub

## Structure du dépôt

```
├── notebooks/          # Exploration, préparation, baseline, SFT, test réaliste, DPO (exécutés sur Kaggle)
├── src/
│   ├── cleaning.py     # Nettoyage et échantillonnage des données
│   ├── prompt.py       # Consigne et format des exemples, communs à l'entraînement et à l'évaluation
│   ├── generation.py   # Génération des anonymisations par lots
│   ├── evaluation.py   # Métriques : protection, préservation, fidélité, correspondance exacte
│   ├── annotation.py   # Construction d'exemples annotés (jeu de test réaliste)
│   └── dpo.py          # Génération des textes ciblés et des paires de préférences
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
