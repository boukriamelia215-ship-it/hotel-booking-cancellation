# 🏨 Prédiction des Annulations Hôtelières

## Contexte du projet
Dans le secteur hôtelier, les annulations de dernière minute représentent 
une perte importante de revenus. Une chambre annulée au dernier moment 
reste vide et génère une perte sèche pour l'établissement.

Ce projet vise à construire un modèle capable de prédire si une réservation 
va être annulée, afin d'anticiper ces pertes et optimiser le taux d'occupation.

## Dataset
- **Source** : Kaggle - Hotel Booking Cancellations
- **Lien** : https://www.kaggle.com/datasets/muhammaddawood42/hotel-booking-cancelations
- **Volume** : 119 390 réservations, 36 colonnes
- **Variable cible** : is_canceled (0 = non annulé, 1 = annulé)

## Structure du projet
```
hotel-booking-cancellation/
├── EDA.ipynb            # Notebook complet : EDA, nettoyage, modélisation
├── README.md            # Ce fichier
└── API/
    ├── app.py           # API Flask avec 4 endpoints
    ├── model.pkl        # Modèle Random Forest sauvegardé
    ├── kmeans.pkl       # Modèle KMeans sauvegardé
    └── requirements.txt # Dépendances Python
```

## Ce qu'on a fait

### 1. Exploration des données (EDA)
- Analyse de la structure et des statistiques descriptives
- Détection et traitement des valeurs manquantes
- Détection des valeurs aberrantes avec boxplots
- Matrice de corrélation et V de Cramer
- Hypothèses métier validées visuellement

### 2. Nettoyage et Preprocessing
- Suppression des colonnes inutiles (données personnelles, data leakage)
- Traitement des valeurs manquantes selon le contexte
- Label Encoding des variables catégorielles
- Standardisation avec StandardScaler

### 3. Modélisation supervisée
| Modèle | Accuracy |
|--------|----------|
| Random Forest | 89.4% |
| Decision Tree | 85.6% |
| Régression Logistique | 79.9% |

On a choisi le **Random Forest** comme modèle final car il donne les meilleures performances.

### 4. Modélisation non supervisée
- KMeans avec 4 clusters identifiés via la méthode du coude
- Segmentation des clients en 4 profils distincts

## Lancer l'API

### Installation
```
pip install -r requirements.txt
```

### Démarrage
```
cd API
python app.py
```

### Endpoints disponibles

**POST /predict** → Prédiction binaire
```
{"features": [valeurs...]}
→ {"prediction": 1, "resultat": "Annule"}
```

**POST /predict_proba** → Probabilité d'annulation
```
{"features": [valeurs...]}
→ {"probabilite_annulation": 69.0, "probabilite_non_annulation": 31.0}
```

**POST /predict_cluster** → Segment client
```
{"features": [valeurs...]}
→ {"cluster": 3, "description": "..."}
```

**GET /clusters_summary** → Infos sur les segments
```
→ {"cluster_0": "...", "cluster_1": "...", ...}
```

## Résultats et impact business
Le modèle détecte 82% des annulations réelles avec 89.4% d'accuracy globale.
Cela permet à l'hôtel d'anticiper les chambres libres et de contacter 
les clients à risque avant qu'ils annulent.
