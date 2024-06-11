import requests
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    data = geocode()
    lat = data["items"][0]["position"]["lat"]
    lng = data["items"][0]["position"]["lng"]
    return render_template('index.html', lat=lat, lng=lng)

def geocode():
    api_key = "g5pnTzNpvTx0Uo_tTMCY9PW8mLsMpfMpotYavOUeuHA"
    param = "Porto"
    url = f"https://geocode.search.hereapi.com/v1/geocode?q={param}&apiKey={api_key}"
    response = requests.get(url)
    return response.json()
