# Importation de pandas
import pandas as pd
import requests
import gensim
import spacy
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

def tokenisation(file):
    print("On lance la tokenisation")
    print(file)
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