# Documentation de l'API

## API de prédiction d'adresses

### Présentation
Cette API permet de prédire des adresses de la ville de Paris. Elle offre la possibilité d'entraîner un modèle via l'import d'un fichier csv et de converser avec un assistant sur des lieux.

### Endpoints

#### Entrainement du modèle

##### `POST /training`
**Description** : Entraîne le modèle à partir d'un fichier CSV déversé par l'utilisateur.

**Paramètres** :
- `file` : Fichier CSV contenant les données d'entraînement.

**Réponse** :
- `200 OK` : Modèle entraîné et sauvegardé avec succès. Retourne un message de succès et le chemin du graphique d'entraînement.
- `400 Bad Request` : Erreur lors de l'entraînement du modèle. Retourne le détail de l'erreur.

---

#### Prédiction d'émotion

##### `POST /predict`
**Description** : Prédit les émotions à partir d'une image fournie.

**Paramètres** :
- `file` : Image capturé d'une personne dont l'émotion doit être prédite.

**Réponse**:
- `200 OK` : Prédiction réalisée avec succès. Retourne l'émotion prédite.
- `500 Internal Server Error` : Erreur lors de la prédiction. Retourne le détail de l'erreur.

---

#### OpenAI

##### `GET /model`
**Description** : Traite un texte fourni en utilisant des techniques de tokenisation et de réponse via l'API OpenAI basé sur un pré-entrainement.

**Paramètres**:

- `text`: Texte à traiter.

**Réponse**:

- `200 OK`: Retourne les tokens générés et une répponse basée sur le texte fourni.
- `500 Internal Server Error`: Erreur lors du traitement du texte. Retourne le détail de l'erreur.