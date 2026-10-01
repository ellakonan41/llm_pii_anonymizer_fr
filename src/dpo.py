"""Construction des paires de préférences pour le DPO.

Les textes sont générés à partir de gabarits ciblant les faiblesses observées
sur le jeu de test réaliste (heures et dates au format français, noms de rue
composés de mots courants, fidélité de la recopie). Les listes de valeurs sont
distinctes de celles du jeu de test réaliste.

La réponse rejetée est l'erreur du modèle lui-même quand il se trompe, sinon
une erreur construite à partir de la bonne réponse.
"""

import random
import re
import string
import unicodedata

from src.annotation import construire_exemple
from src.evaluation import evaluer_exemple
from src.prompt import construire_messages

PRENOMS = [
    "Manon", "Louis", "Jade", "Gabriel", "Lina", "Arthur", "Zoé", "Nathan", "Clara", "Raphaël",
    "Lola", "Adam", "Louise", "Ethan", "Alice", "Mohamed", "Juliette", "Noah", "Tom", "Lucie",
    "Maxime", "Pauline", "Quentin", "Margaux", "Victor", "Eva", "Samuel", "Agathe", "Florian",
    "Rayan", "Yasmine", "Baptiste", "Charlotte", "Mathis", "Elsa", "Théo", "Salomé", "Clément",
    "Laurent", "Ibrahim", "Awa", "Moussa", "Mei", "Sacha",
    # Prénoms qui sont aussi des noms communs
    "Violette", "Ambre", "Victoire", "Constance", "Prudence", "Capucine", "Aimé", "Désiré",
]
NOMS = [
    "Durand", "Leroux", "Mallet", "Perrot", "Colin", "Arnaud", "Giraud", "Barbier", "Meunier",
    "Boyer", "Gautier", "Masson", "Picard", "Fournier", "Rey", "Lacroix", "Marchal", "Hamon",
    "Leblanc", "Joly", "Collet", "Bailly", "Fleury", "Huet", "Tessier", "Mathieu", "Dupuis",
    "Diop", "Koné", "Ouattara", "Tran", "Lefort", "Morin", "Chauvin", "Poulain", "Benhamou",
    "Le Bihan", "Da Silva",
    # Noms qui sont aussi des noms communs
    "Renard", "Noël", "Carré", "Boucher", "Charpentier", "Berger", "Rivière", "Meyer",
]
VILLES = [
    "Nancy", "Brest", "Angers", "Reims", "Metz", "Tours", "Limoges", "Caen", "Rouen", "Orléans",
    "Clermont-Ferrand", "Avignon", "Pau", "Annecy", "Poitiers", "Amiens", "Besançon",
    "Perpignan", "Nîmes", "Le Mans", "Vannes", "Colmar", "Valence", "Saint-Malo",
]
RUES = [
    # Noms de rue composés de mots courants
    "rue de la Liberté", "place de l'Église", "avenue de la Gare", "rue des Écoles",
    "chemin des Vignes", "allée des Tilleuls", "impasse du Château", "rue de la Paix",
    "avenue des Fleurs", "rue du Marché", "boulevard de la Plage", "rue des Jardins",
    "place du Marché", "rue de la Fontaine", "route du Lac", "rue du Stade", "rue de la Poste",
    "chemin de la Forêt", "rue des Prés", "quai des Pêcheurs",
    # Noms de rue tirés de noms de personnes
    "rue Victor Hugo", "avenue Pasteur", "rue Émile Zola", "boulevard Gambetta",
    "rue Jean Moulin", "avenue Foch", "rue Gustave Eiffel", "place Charles de Gaulle",
]
TITRES = ["Monsieur", "Madame", "M.", "Mme"]
DOCTEURS = ["Dr", "Docteur"]
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"]
MOIS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]
DOMAINES = ["gmail.com", "orange.fr", "free.fr", "laposte.net", "hotmail.fr", "yahoo.fr"]

# Champ de gabarit -> étiquette (« jour » n'est pas une donnée personnelle).
ETIQUETTE_DU_CHAMP = {
    "prenom": "GIVENNAME", "nom": "SURNAME", "titre": "TITLE", "dr": "TITLE",
    "heure": "TIME", "date": "DATE", "numero": "BUILDINGNUM", "rue": "STREET",
    "cp": "ZIPCODE", "ville": "CITY", "tel": "TELEPHONENUM", "email": "EMAIL", "age": "AGE",
}

GABARITS = [
    "Bonjour {titre} {nom}, votre rendez-vous est confirmé le {jour} {date} à {heure}. Merci de vous présenter en avance.",
    "Rappel : réunion d'équipe {jour} à {heure} en salle de conférence. {prenom}, peux-tu préparer la présentation ?",
    "Livraison prévue le {date} entre {heure} et {heure2} au {numero} {rue}, {cp} {ville}.",
    "Merci d'envoyer le courrier à {prenom} {nom}, {numero} {rue}, {cp} {ville}.",
    "Le cabinet du {dr} {nom} a déménagé au {numero} {rue} à {ville}. Nouveau numéro : {tel}.",
    "{prenom} {nom} sera absent du {date} au {date2}. Pour toute urgence, contactez {prenom2} au {tel}.",
    "Votre colis vous attend au point relais situé {numero} {rue}. Ouverture du {jour} au samedi, de {heure} à {heure2}.",
    "Inscription de {prenom}, {age} ans, à l'atelier du {jour} {date}, de {heure} à {heure2}.",
    "Je soussigné {titre} {prenom} {nom}, demeurant {numero} {rue}, {cp} {ville}, atteste sur l'honneur l'exactitude des informations fournies.",
    "Le départ du car est fixé à {heure} le {date}, devant la mairie de {ville}.",
    "[{heure}] {prenom} : je suis en retard, j'arrive vers {heure2}.",
    "Bonjour, je vous contacte au sujet de la facture du {date}. Vous pouvez me joindre au {tel} après {heure}. {prenom} {nom}",
    "Rendez-vous chez le notaire, {numero} {rue}, le {jour} {date} à {heure}. Merci de venir avec une pièce d'identité.",
    "Le marché de {ville} se tient {rue} chaque {jour} matin.",
    "Ma nouvelle adresse : {numero} {rue}, {cp} {ville}. Mon e-mail ne change pas : {email}.",
    "Le {date}, {titre} {nom} a signalé une fuite d'eau dans son logement du {numero} {rue}.",
    "Salut {prenom} ! On se retrouve {jour} vers {heure} devant le {numero} {rue} ?",
    "Consultation du {date} : le patient, {age} ans, sera revu par le {dr} {nom} dans trois semaines.",
    "Merci de rappeler {titre} {nom} au {tel} avant {heure} pour confirmer l'intervention du {date}.",
    "Visite de l'appartement du {numero} {rue} prévue {jour} {date} à {heure}. Contact agence : {prenom} {nom}, {tel}.",
]

PHRASES_SANS_DONNEE = [
    "Les inscriptions pour les activités périscolaires sont ouvertes jusqu'à la fin du mois.",
    "Merci de bien vouloir fermer les fenêtres en quittant les locaux.",
    "La cantine proposera un menu végétarien chaque semaine à partir de la rentrée.",
    "Pensez à sauvegarder régulièrement vos documents sur le serveur partagé.",
    "Le parking souterrain sera fermé pour travaux pendant deux semaines.",
    "Les formulaires de remboursement doivent être accompagnés des justificatifs originaux.",
    "Une coupure d'électricité est prévue dans tout le bâtiment pour maintenance.",
    "Quelles pièces justificatives faut-il fournir pour une demande d'aide au logement ?",
    "La bibliothèque municipale prolonge ses horaires d'ouverture pendant les examens.",
    "Les résultats de l'enquête de satisfaction seront présentés lors du prochain comité.",
    "Merci de signaler toute anomalie constatée sur les équipements informatiques.",
    "Le règlement intérieur a été mis à jour et doit être signé par chaque collaborateur.",
]


def _sans_accents(texte: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn")


def _heure(rng: random.Random) -> str:
    h, m = rng.randint(7, 20), rng.choice([0, 15, 30, 45, 5, 10, 50])
    return rng.choice([f"{h}h{m:02d}", f"{h}h{m:02d}", f"{h}h", f"{h}:{m:02d}", f"{h} h {m:02d}"])


def _date(rng: random.Random) -> str:
    j, m, a = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2023, 2026)
    jour = "1er" if j == 1 else str(j)
    return rng.choice([f"{jour} {MOIS[m - 1]}", f"{jour} {MOIS[m - 1]} {a}", f"{j:02d}/{m:02d}/{a}"])


def _tel(rng: random.Random) -> str:
    chiffres = [rng.choice("67")] + [f"{rng.randint(0, 99):02d}" for _ in range(4)]
    paires = [f"0{chiffres[0]}"] + chiffres[1:]
    return rng.choice([
        " ".join(paires), ".".join(paires), "".join(paires),
        f"+33 {chiffres[0]} " + " ".join(chiffres[1:]),
    ])


def _valeur(champ: str, rng: random.Random, valeurs: dict) -> str:
    if champ == "prenom":
        return rng.choice(PRENOMS)
    if champ == "nom":
        return rng.choice(NOMS)
    if champ == "titre":
        return rng.choice(TITRES)
    if champ == "dr":
        return rng.choice(DOCTEURS)
    if champ == "heure":
        return _heure(rng)
    if champ == "date":
        return _date(rng)
    if champ == "jour":
        return rng.choice(JOURS)
    if champ == "numero":
        return str(rng.randint(1, 150))
    if champ == "rue":
        return rng.choice(RUES)
    if champ == "cp":
        return f"{rng.randint(1, 95):02d}{rng.randint(0, 9)}{rng.randint(0, 9)}0"
    if champ == "ville":
        return rng.choice(VILLES)
    if champ == "tel":
        return _tel(rng)
    if champ == "age":
        return str(rng.randint(18, 90))
    if champ == "email":
        prenom = valeurs.get("prenom", rng.choice(PRENOMS))
        nom = valeurs.get("nom", rng.choice(NOMS))
        local = _sans_accents(f"{prenom}.{nom}").lower().replace(" ", "")
        return f"{local}@{rng.choice(DOMAINES)}"
    raise ValueError(f"Champ inconnu : {champ}")


def remplir_gabarit(gabarit: str, rng: random.Random) -> tuple[str, list[list[str]]]:
    """Remplit un gabarit et renvoie le texte et ses entités [étiquette, valeur]."""
    champs = [nom for _, nom, _, _ in string.Formatter().parse(gabarit) if nom]
    # L'e-mail est construit à partir du prénom et du nom : on le remplit en dernier.
    champs.sort(key=lambda c: c.startswith("email"))
    valeurs, entites = {}, []
    for champ in champs:
        base = re.sub(r"\d+$", "", champ)
        valeurs[champ] = _valeur(base, rng, valeurs)
        if base in ETIQUETTE_DU_CHAMP:
            entite = [ETIQUETTE_DU_CHAMP[base], valeurs[champ]]
            if entite not in entites:
                entites.append(entite)
    return gabarit.format(**valeurs), entites


def generer_exemples(n: int, graine: int = 0, part_sans_donnee: float = 0.1) -> list[dict]:
    """Génère n exemples annotés, de façon reproductible."""
    rng = random.Random(graine)
    exemples = []
    while len(exemples) < n:
        uid = f"dpo-{len(exemples):04d}"
        if rng.random() < part_sans_donnee:
            exemples.append(construire_exemple(rng.choice(PHRASES_SANS_DONNEE), [], uid=uid))
            continue
        texte, entites = remplir_gabarit(rng.choice(GABARITS), rng)
        try:
            exemples.append(construire_exemple(texte, entites, uid=uid))
        except ValueError:
            # Valeurs qui se chevauchent (ex. un numéro contenu dans une date) : on retire.
            continue
    return exemples


def anonymiser_sauf(exemple: dict, indices_en_clair: set[int]) -> str:
    """Texte anonymisé de référence, en laissant en clair les entités désignées."""
    texte = exemple["source_text"]
    entites = sorted(enumerate(exemple["privacy_mask"]), key=lambda p: p[1]["start"], reverse=True)
    for i, e in entites:
        if i not in indices_en_clair:
            texte = texte[: e["start"]] + f"[{e['label']}]" + texte[e["end"] :]
    return texte


# Mots d'au moins 5 lettres, hors étiquettes entre crochets.
_MOT_OU_ETIQUETTE = re.compile(r"\[[A-Z]+\]|[^\W\d_]{5,}")


def _mots_modifiables(texte: str) -> list[re.Match]:
    return [m for m in _MOT_OU_ETIQUETTE.finditer(texte) if not m.group().startswith("[")]


def introduire_faute(texte: str, rng: random.Random) -> str | None:
    """Supprime une lettre au milieu d'un mot (erreur de recopie)."""
    mots = [m for m in _mots_modifiables(texte) if len(m.group()) >= 6]
    if not mots:
        return None
    m = rng.choice(mots)
    i = rng.randint(1, len(m.group()) - 2)
    return texte[: m.start() + i] + texte[m.start() + i + 1 :]


def masquer_en_trop(texte: str, rng: random.Random) -> str | None:
    """Remplace un mot ordinaire par une étiquette (masquage injustifié)."""
    mots = _mots_modifiables(texte)
    if not mots:
        return None
    m = rng.choice(mots)
    etiquette = rng.choice(["CITY", "SURNAME", "GIVENNAME", "STREET"])
    return texte[: m.start()] + f"[{etiquette}]" + texte[m.end() :]


# Types les plus souvent laissés en clair sur le jeu de test réaliste.
TYPES_CIBLES = {"TIME", "DATE", "STREET"}


def corrompre(exemple: dict, rng: random.Random) -> tuple[str, str] | None:
    """Construit une mauvaise réponse à partir de la bonne : fuite, faute ou masquage en trop."""
    reference = exemple["masked_text"]
    strategies = ["faute", "en_trop"]
    if exemple["privacy_mask"]:
        strategies = ["fuite"] * 3 + strategies
    rng.shuffle(strategies)
    for strategie in strategies:
        if strategie == "fuite":
            poids = [3 if e["label"] in TYPES_CIBLES else 1 for e in exemple["privacy_mask"]]
            i = rng.choices(range(len(exemple["privacy_mask"])), weights=poids)[0]
            rejet = anonymiser_sauf(exemple, {i})
        elif strategie == "faute":
            rejet = introduire_faute(reference, rng)
        else:
            rejet = masquer_en_trop(reference, rng)
        if rejet is not None and rejet != reference:
            return rejet, strategie
    return None


def choisir_rejet(exemple: dict, sortie_modele: str | None, rng: random.Random) -> tuple[str, str] | None:
    """Réponse rejetée : l'erreur du modèle s'il en a fait une, sinon une erreur construite."""
    if sortie_modele is not None and sortie_modele.strip() != exemple["masked_text"].strip():
        r = evaluer_exemple(exemple, sortie_modele)
        if r["nb_protegees"] < r["nb_entites"] or r["nb_mots_conserves"] < r["nb_mots_attendus"]:
            return sortie_modele, "modele"
    return corrompre(exemple, rng)


def paire_dpo(exemple: dict, rejet: str) -> dict:
    """Paire de préférences au format conversationnel attendu par le DPOTrainer de trl."""
    return {
        "prompt": construire_messages(exemple["source_text"]),
        "chosen": [{"role": "assistant", "content": exemple["masked_text"]}],
        "rejected": [{"role": "assistant", "content": rejet}],
    }
