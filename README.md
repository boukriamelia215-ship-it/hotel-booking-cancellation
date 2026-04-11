# Projet Data Science — Prédiction des annulations hôtelières

## Problématique métier

Dans le secteur hôtelier les annulations de dernière minute représentent un problème
économique réel. Une chambre annulée au dernier moment reste vide et l'hôtel perd
directement de l'argent. Sur notre dataset on constate que 37% des réservations
finissent par être annulées ce qui est considérable.

L'objectif de ce projet est de prédire si une réservation va être annulée avant
l'arrivée du client. Avec ce modèle l'hôtel peut anticiper les annulations,
contacter les clients à risque, mieux gérer ses chambres disponibles et proposer
des offres de rétention ciblées.

## Dataset

Source : Kaggle - Hotel Booking Cancelations
Lien : https://www.kaggle.com/datasets/muhammaddawood42/hotel-booking-cancelations
Volume : 119 390 réservations réelles entre 2015 et 2017
Nombre de variables : 36 colonnes dont le lead_time, le type de dépôt, le pays
du client, le nombre de nuits, les demandes spéciales, le prix par nuit, etc.

## Analyse exploratoire des données

On a commencé par analyser la structure du dataset, les types de colonnes et
les statistiques descriptives. On a ensuite détecté les valeurs manquantes :
la colonne company avait 94% de valeurs manquantes donc on l'a supprimée.
Les colonnes children, agent et country ont été traitées selon leur contexte.

On a utilisé des boxplots pour détecter les outliers sur lead_time et adr.
On a calculé la matrice de corrélation pour les variables numériques et le
V de Cramér pour analyser les liens entre variables catégorielles.

Deux hypothèses métier ont été formulées et vérifiées avec des graphiques :
la première est que le lead_time influence les annulations (confirmée, les
réservations annulées ont un lead_time moyen deux fois plus élevé),
la deuxième est que le type de dépôt influence les annulations (confirmée,
les réservations sans dépôt sont beaucoup plus souvent annulées).

## Nettoyage et preprocessing

On a supprimé les colonnes inutiles : company pour les valeurs manquantes,
name, email, phone-number et credit_card pour les données personnelles, et
reservation_status car elle donne directement la réponse et fausserait le modèle.

Pour l'encodage on a choisi le Label Encoding plutôt que le One Hot Encoding
car la colonne country a 177 valeurs distinctes et le One Hot aurait ajouté
177 colonnes ce qui aurait rendu le modèle beaucoup plus lent.

On a appliqué une transformation log1p sur les colonnes très asymétriques pour
réduire l'effet des valeurs extrêmes. On a ensuite séparé le dataset en 80% train
et 20% test et normalisé avec le StandardScaler.

Au final on a perdu seulement 149 lignes sur 119 390 soit 0.12% du dataset.

## Modélisation supervisée

On a entraîné et comparé trois modèles :

Random Forest : accuracy 89.4% et AUC 0.96 — meilleur modèle retenu
Decision Tree : accuracy 85.6% et AUC 0.847
Régression Logistique : accuracy 79.9% et AUC 0.761

Le Random Forest a été choisi car il obtient les meilleures performances sur
toutes les métriques. Les variables les plus importantes sont le lead_time et le
deposit_type ce qui confirme nos hypothèses formulées en début d'analyse.

## Modélisation non supervisée

On a appliqué un KMeans pour segmenter les clients en groupes homogènes sans
utiliser la variable cible. La méthode du coude a permis de choisir 4 clusters.

Cluster 0 : 12 654 réservations, profil atypique, probablement des groupes ou clients corporate
Cluster 1 : 46 299 réservations, profil majoritaire, clients standard
Cluster 2 : 19 745 réservations, clients qui réservent très longtemps à l'avance
Cluster 3 : 16 694 réservations, clients avec beaucoup de demandes spéciales

Ces segments permettent à l'hôtel d'adapter sa stratégie commerciale selon
le profil de chaque groupe de clients.

## Sauvegarde des modèles

Les deux modèles ont été sauvegardés au format .pkl et peuvent être rechargés
sans avoir besoin de les réentraîner.

model.pkl : modèle Random Forest
kmeans.pkl : modèle KMeans

## API Flask

Les modèles sont exposés via une API Flask avec les endpoints suivants :

GET  /                 : page d'accueil
POST /predict          : prédiction binaire, renvoie annulé ou non annulé
POST /predict_proba    : probabilité d'annulation en pourcentage
POST /predict_cluster  : cluster auquel appartient la réservation
GET  /clusters_summary : résumé des 4 clusters

## Lancer le projet

Sans Docker :
cd API
pip install -r requirements.txt
python app.py

Avec Docker :
cd API
docker build -t hotel-api .
docker run -p 5000:5000 hotel-api

L'API est accessible sur http://127.0.0.1:5000

## Analyse business et impact

Avec une accuracy de 89.4% le modèle permet à l'hôtel d'identifier les
réservations à risque avant l'arrivée du client. Cela lui permet de contacter
ces clients pour confirmer, de mieux gérer les overbookings et de proposer
des offres de rétention adaptées.

Les limites du modèle : les données datent de 2015 à 2017 donc il pourrait
être moins performant sur des comportements clients plus récents. Le Label
Encoding sur country peut aussi introduire un biais.

Pour aller plus loin on pourrait tester XGBoost, ajouter des variables externes
comme la météo ou les événements locaux, et mettre en place une alerte
automatique quand un client est détecté à risque.
