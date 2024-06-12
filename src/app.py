import requests
from flask import Flask, render_template, jsonify, request
from gensim.models import Word2Vec


app = Flask(__name__)
model = Word2Vec.load("src/static/word2vec_france.model")  

@app.route("/")
def home():
    data = geocode()
    lat = data["items"][0]["position"]["lat"]
    lng = data["items"][0]["position"]["lng"]
    pred = predict("omer")
    return render_template('index.html', lat=lat, lng=lng, pred=pred)

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