import pickle
import pandas as pd
import joblib
import numpy as np
import shap 
from zipfile import ZipFile

from flask import Flask, render_template, request
app = Flask(__name__)

# Charger le modèle pickle
model = joblib.load("LGBM_Classifier_final_v2.pkl")
# Chargement de l'Explainer SHAP
with open('explainer_v2.pkl', 'rb') as file:
    explainer = pickle.load(file)

# Charger l'ensemble de données depuis un fichier CSV
z1 = ZipFile("X_data_test.zip")
data = pd.read_csv(z1.open('X_data_test.csv'), encoding ='utf-8')
#data = data.drop(["TARGET"], axis=1)
@app.route("/")
def home():
    return "Bienvenue sur l'application Flask !"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Récupérer l'identifiant du client depuis le formulaire
        client_id = int(request.form["client_id"])
    elif request.method == "GET":
        # Récupérer l'identifiant du client depuis l'URL
        client_id = int(request.args.get("client_id"))

    # Vérifier si l'identifiant du client existe dans l'ensemble de données
    if client_id not in data["SK_ID_CURR"].values:
        return f"L'identifiant client {client_id} n'existe pas dans l'ensemble de données."

    # Filtrer les données pour le client spécifié
    client_data = data[data["SK_ID_CURR"] == client_id]

    # Extraire les caractéristiques du client
    features = client_data.drop(["SK_ID_CURR"], axis=1)

    # Effectuer la prédiction en utilisant le modèle chargé
    y_pred = model.predict_proba(features)

    # Calculer le score  à partir de y_true et y_pred
    score_metier = custom_score(y_pred[:, 1])  # Appeler votre fonction de calcul du score métier

    # Renvoyer le score métier en tant que réponse
    return score_metier
##################################################################################################################
@app.route("/score", methods=["GET", "POST"])
def score():
    if request.method == "POST":
        # Récupérer l'identifiant du client depuis le formulaire
        client_id = int(request.form["client_id"])
    elif request.method == "GET":
        # Récupérer l'identifiant du client depuis l'URL
        client_id = int(request.args.get("client_id"))

    # Vérifier si l'identifiant du client existe dans l'ensemble de données
    if client_id not in data["SK_ID_CURR"].values:
        return f"L'identifiant client {client_id} n'existe pas dans l'ensemble de données."

    # Filtrer les données pour le client spécifié
    client_data = data[data["SK_ID_CURR"] == client_id]

    # Extraire les caractéristiques du client
    features = client_data.drop(["SK_ID_CURR"], axis=1)

    # Effectuer la prédiction en utilisant le modèle chargé
    y_pred = model.predict_proba(features)

    # Calculer le score à partir de y_true et y_pred
    score_metier = pourcentage(y_pred[:, 1])  # Appeler votre fonction de calcul du score métier

    # Renvoyer le score métier en tant que réponse
    return score_metier



###################################################################################################################



@app.route('/feature_importance', methods=['POST'])
def calculate_feature_importance():
    data = request.get_json()
    
        
    # Convertir les données en DataFrame
    df = pd.DataFrame(data)

    shap_values = explainer.shap_values(df)
    # Calculer les importances moyennes pour chaque feature
    feature_importances = pd.DataFrame(shap_values, columns=df.columns).abs().mean(axis=0)
    
    # Convertir les importances en dictionnaire
    importance_dict = feature_importances.to_dict()
    
    # Renvoyer le dictionnaire de résultats en tant que réponse JSON
    return jsonify(importance_dict)





def pourcentage(proba):
    score = proba * 100
    score_rounded = round(score, 2)
    return score_rounded



def custom_score(y_pred):
    # Implémentez votre propre logique pour calculer le score métier
    # Utilisez les valeurs de y_true et y_pred pour effectuer les calculs nécessaires
    # Par exemple :
    if y_pred >= 0.3 and y_pred <0.4:        
        score = 3
    elif y_pred >= 0.4:
        score = 1
    else:
        score = 0
    return str(score)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


