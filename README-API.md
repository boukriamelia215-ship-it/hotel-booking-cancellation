# Prédiction des annulations hôtelières

## Contexte

Ce projet a été réalisé dans le cadre du cours de Data Science avec Python.
L'objectif est de prédire si une réservation hôtelière va être annulée ou non,
à partir d'un dataset Kaggle contenant plus de 119 000 réservations réelles.

## Structure du projet

API/
├── app.py             # API Flask
├── model.pkl          # Modèle Random Forest sauvegardé
├── kmeans.pkl         # Modèle KMeans sauvegardé
├── requirements.txt   # Dépendances Python
└── .venv/             # Environnement virtuel

## Installation

Cloner le projet puis installer les dépendances :

pip install -r requirements.txt

## Lancer l'API

python app.py

L'API tourne sur http://127.0.0.1:5000

## Endpoints disponibles

Méthode | Endpoint          | Description
GET     | /                 | Page d'accueil
POST    | /predict          | Prédiction binaire (annulé ou non)
POST    | /predict_proba    | Probabilité d'annulation
POST    | /predict_cluster  | Cluster de la réservation
GET     | /clusters_summary | Résumé des 4 clusters

## Exemple d'utilisation

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [0,1,2,0,0,0,0,2,0,1,1,0,0,0,0,1.099,0,0,0,0,0,0,0,0,1,0,0,0,0]}'

## Modèles utilisés

Random Forest (supervisé) : accuracy de 89.4%, AUC de 0.96
KMeans (non supervisé) : 4 clusters de clients identifiés

## Dataset

Source : Kaggle - Hotel Booking Demand
Lien : https://www.kaggle.com/datasets/muhammaddawood42/hotel-booking-cancelations
Volume : 119 390 réservations, 36 variables