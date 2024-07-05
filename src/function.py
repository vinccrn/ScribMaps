# Importation de pandas
import pandas as pd
import requests
import gensim
import spacy
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import subprocess
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from openai import OpenAI

nltk.download('stopwords')
nltk.download('punkt')

openai_client = OpenAI(api_key="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")

stop_words = set(stopwords.words('french'))


def install_spacy_model():
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
        print("Le modèle fr_core_news_sm a été installé avec succès.")
    except subprocess.CalledProcessError:
        print("Une erreur s'est produite lors de l'installation du modèle.")

def tokenisation(file):
    print("Vérification du Spacy Model pour la tokenisation NLP")
    print("-------------------------------")
    install_spacy_model()
    print("lancement de la tokenisation...")
    print("-------------------------------")
    print("Ladies and gentlemen, nous sommes sur le point d'atterir. Veuillez patienter environ 10 Minutes pour que le traitement NLP se termine")

    df = pd.read_csv(file, delimiter=";")
    df_france = df.drop(['id', 'id_fantoir', 'code_insee', 'code_insee_ancienne_commune', 'nom_ancienne_commune', 'x', 'y', 'lat', 'lon', 'type_position', 'alias', 'nom_ld', 'libelle_acheminement',  'nom_afnor', 'source_position', 'source_nom_voie', 'certification_commune', 'cad_parcelles'], axis=1)
    df_france = df_france.assign(pays="France")
    df_france.to_csv('data_france.txt', sep='\t', index=False)
    
    with open(r'data_france.txt', 'r') as file:
        data = file.read()
        data = data.replace("		", " ")

    with open(r'data_france.txt', 'w') as file:
        file.write(data)

    nlp = spacy.load("fr_core_news_sm")  

    # Tokenisation
    tokenized_data = list()

    with open(r'data_france.txt', 'r') as file:
        data = file.read()

        lowercase = data.lower()

        lines = lowercase.splitlines()

        for line in lines:
            tokens = nlp(line)
            #mots = [token.text for token in tokens]
            mots = [token.text for token in tokens if token.text != '\t']
            tokenized_data.append(mots)
    print("fin de la tokenisation...")
    print("-------------------------------")
    return tokenized_data

def entrainement_modele(file):
    tokenized_data = tokenisation(file)

    model = gensim.models.Word2Vec(min_count=1, window=5, sg=1)

    model.build_vocab(tokenized_data)
    
    model.train(tokenized_data, total_examples=model.corpus_count, epochs=10)
    #Sauvegarde du model
    model.save("word2vec_france.model")

def predire_adresse(saisie, adresses, seuil_similarite=0.90):
    model = Word2Vec.load("src/static/word2vec_france.model")  

    saisie_tokens = saisie.lower().split()  
    print(saisie_tokens)
    meilleure_adresse = None
    meilleure_similarite = 0

    for adresse in adresses:
        if len(adresse) > len(saisie_tokens):  # Vérifier si l'adresse est plus longue que la saisie
            similarite = model.wv.n_similarity(saisie_tokens, adresse[:len(saisie_tokens)])
            #print(similarite)
            if similarite > meilleure_similarite and similarite >= seuil_similarite:
                meilleure_suite = " ".join(adresse[len(saisie_tokens):])  # Récupérer la suite de l'adresse
                print(meilleure_suite)
                meilleure_similarite = similarite
                print(meilleure_similarite)

    return meilleure_adresse

def nlp_tokenization(document: str):
    tokens = word_tokenize(document)
    return [token for token in tokens if token.lower() not in stop_words]

def question_with_gpt3(text: str, adresse_predite: str):
    try:
        prompt = (
            "Vous êtes un assistant spécialisé dans l'analyse d'adresses. "
            f"L'utilisateur a saisi : \"{text}\". "
            f"L'adresse prédite est : \"{adresse_predite}\". "
            "Veuillez fournir des informations pertinentes sur cette adresse ou des recommandations utiles. "
            "Si l'adresse prédite est vide ou non pertinente, donnez des conseils généraux sur la recherche d'adresses."
        )
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Analyse de l'adresse : {adresse_predite}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erreur lors de l'appel à GPT-3: {str(e)}")
        return "Erreur système lors de l'analyse de l'adresse."