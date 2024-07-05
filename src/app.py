import requests
from flask import Flask, render_template, jsonify, request
from gensim.models import Word2Vec
import os
from werkzeug.utils import secure_filename

import function  # Sans l'extension .py

# Const
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'upload')#'src/static/upload'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

# Flask 
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tokenisation au démarrage de lappli
files = os.listdir(UPLOAD_FOLDER)

latest_file = max([os.path.join(UPLOAD_FOLDER, f) for f in files], key=os.path.getctime)
print("Pour accéder au serveur sur le port 5000, veuillez patienter la fin de la tokenisation 10 mins environ")
token = function.tokenisation(latest_file)

# App
@app.route("/")
def home():
    data = geocode()
    lat = data["items"][0]["position"]["lat"]
    lng = data["items"][0]["position"]["lng"]
    pred = predict("omer")
    return render_template('index.html', lat=lat, lng=lng, pred=pred)

@app.route("/test")
def test():
    return render_template('test.html')

def geocode():
    api_key = "g5pnTzNpvTx0Uo_tTMCY9PW8mLsMpfMpotYavOUeuHA"
    param = "Porto"
    url = f"https://geocode.search.hereapi.com/v1/geocode?q={param}&apiKey={api_key}"
    response = requests.get(url)
    return response.json()

def predict(token):
    if token in model.wv:
        similar_words = model.wv.most_similar(token)
        return jsonify(similar_words)
    else:
        return jsonify({"error": "Mot non trouvé dans le vocabulaire"}), 404

# API

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/scribmaps/upload/", methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'myfile' not in request.files:
            return jsonify({"error": "Aucun fichier n'a été envoyé"}), 400

        file = request.files['myfile']
        if file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_path = os.path.join(UPLOAD_FOLDER, filename)
            print(f"Trying to save file to: {full_path}")
            file.save(full_path)
            print(file)
            #function.entrainement_modele(full_path)
            return jsonify({"message": "Fichier uploadé avec succès"}), 200
        else:
            return jsonify({"error": "Type de fichier non autorisé"}), 400
        
@app.route("/api/scribmaps/start_training/", methods=['POST'])
def start_training():
    files = os.listdir(UPLOAD_FOLDER)
    if not files:
        return jsonify({"error": "Aucun fichier disponible pour l'entraînement"}), 400

    latest_file = max([os.path.join(UPLOAD_FOLDER, f) for f in files], key=os.path.getctime)
    
    try:
        print(latest_file)
        function.entrainement_modele(latest_file)
        return jsonify({"message": "Entraînement lancé avec succès"}), 200
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'entraînement: {str(e)}"}), 500
    
@app.route("/api/scribmaps/predict", methods=['POST'])
def predict_address():
    data = request.json
    if not data or 'saisie' not in data:
        return jsonify({"error": "La saisie est manquante dans la requête"}), 400

    saisie = data['saisie']
    
    try:
        adresse_predite = function.predire_adresse(saisie, token, 0.90)

        if adresse_predite:
            return jsonify({"adresse_predite": adresse_predite}), 200
        else:
            return jsonify({"message": "Aucune adresse correspondante trouvée"}), 404

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction : {str(e)}"}), 500