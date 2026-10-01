# Guide d'annotation du jeu de test réaliste

Le fichier [`data/test_realiste.jsonl`](../data/test_realiste.jsonl) contient des textes français
réalistes, écrits et annotés à la main pour évaluer le modèle hors des données synthétiques
d'entraînement. Une ligne par texte :

```json
{"texte": "Appelez Marie Dupont au 06 12 34 56 78.", "entites": [["GIVENNAME", "Marie"], ["SURNAME", "Dupont"], ["TELEPHONENUM", "06 12 34 56 78"]]}
```

Les positions et le texte anonymisé de référence sont calculés par
[`src/annotation.py`](../src/annotation.py) ; chaque valeur est masquée à toutes ses occurrences.

## Règles

- **Étiquettes** : celles de la consigne du modèle ([`src/prompt.py`](../src/prompt.py)), pour
  l'évaluer sur ce qu'on lui demande.
- **Valeur** : recopiée exactement comme dans le texte (majuscules et accents compris).
- **Noms** : prénom (`GIVENNAME`) et nom (`SURNAME`) toujours séparés.
- **Titres** (`TITLE`) : Monsieur, Madame, Mme, M., Dr, Maître...
- **Adresses** : numéro (`BUILDINGNUM`), voie (`STREET`, type de voie compris : « rue des Tanneurs »),
  code postal (`ZIPCODE`), ville (`CITY`).
- **Dates et heures** : `DATE` (« 14 novembre », « 03/05/1997 »), `TIME` (« 10h30 »).
  Les jours de la semaine et les fêtes (« vendredi », « lundi de Pâques ») ne sont pas annotés.
- **Organisations, lieux publics, professions** : non annotés (hors du périmètre des étiquettes).
- **Une adresse e-mail** est une seule entité `EMAIL`, même si elle contient le nom de la personne.

## Couverture visée (~50 textes)

| Contexte | Exemples de contenu |
|---|---|
| Médical | rendez-vous, compte rendu, ordonnance, carte Vitale |
| RH | candidature, contrat, arrêt maladie, entretien annuel |
| Service client | réclamation, livraison, résiliation, coordonnées |
| Administratif | impôts, mairie, état civil, permis de conduire |
| Banque / assurance | sinistre, carte bancaire, prélèvement |
| Messagerie | SMS, chat d'équipe, e-mail informel |
| Sans donnée personnelle | annonces, consignes, questions générales (~10 %) |

Cas difficiles à inclure : prénoms qui sont aussi des noms communs (Pierre, Rose, Marine, Olive),
noms composés (Jean-Baptiste, Le Goff), formats variés (+33 6..., 06.12.34.56.78), un même nom
répété, texte long avec de nombreuses entités, données en majuscules ou au milieu de chiffres.
