from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# on charge le modele Random Forest
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# on charge le modele KMeans
with open('kmeans.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

@app.route('/')
def home():
    return "API Hotel Booking - Prédiction des annulations"

# endpoint 1 : prédiction binaire (annulé ou pas)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    resultat = "Annule" if prediction[0] == 1 else "Non annule"
    return jsonify({
        'prediction': int(prediction[0]),
        'resultat': resultat
    })

# endpoint 2 : probabilité d'annulation
@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    proba = model.predict_proba(features)
    return jsonify({
        'probabilite_non_annulation': round(proba[0][0] * 100, 2),
        'probabilite_annulation': round(proba[0][1] * 100, 2)
    })

# endpoint 3 : prédiction du cluster d'une réservation
@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    cluster = kmeans_model.predict(features)
    return jsonify({
        'cluster': int(cluster[0]),
        'description': f"Cette réservation appartient au cluster {int(cluster[0])}"
    })

# endpoint 4 : infos sur les clusters
@app.route('/clusters_summary', methods=['GET'])
def clusters_summary():
    return jsonify({
        'cluster_0': '12654 réservations - profil atypique',
        'cluster_1': '46299 réservations - profil majoritaire',
        'cluster_2': '19745 réservations - réservent longtemps à lavance',
        'cluster_3': '16694 réservations - clients avec demandes spéciales'
    })

if __name__ == '__main__':
    app.run(debug=True)