import streamlit as st
from datasets import load_dataset
import spacy
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.spatial.distance import cosine
import nltk
import os
import pickle
from pathlib import Path

# Chemin du fichier cache pour les données nettoyées
CACHE_FILE = Path("cleaned_qa_pairs.pkl")
MODEL_FILE = Path("word2vec_model.pkl")

# Fonction pour sauvegarder les données nettoyées
def save_cleaned_data(qa_pairs):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(qa_pairs, f)

# Fonction pour charger les données nettoyées
def load_cleaned_data():
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)

# Fonction pour sauvegarder le modèle Word2Vec
def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# Fonction pour charger le modèle Word2Vec
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

# Chargement et cache du modèle SpaCy
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

# Chargement du dataset
@st.cache_resource
def load_data():
    dataset = load_dataset("vicgalle/alpaca-gpt4")
    qa_pairs = [
        {"question": item["instruction"], "answer": item["output"]}
        for item in dataset["train"]
    ]
    return qa_pairs

# Nettoyage du texte avec SpaCy (optimisé pour traitement par lot)
def clean_text_batch(texts, nlp):
    docs = list(nlp.pipe(texts))
    return [
        " ".join(
            token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct
        )
        for doc in docs
    ]

# Prétraitement des questions et réponses (optimisé)
def preprocess_qa_pairs(qa_pairs, nlp):
    questions = [pair["question"] for pair in qa_pairs]
    answers = [pair["answer"] for pair in qa_pairs]
    cleaned_questions = clean_text_batch(questions, nlp)
    cleaned_answers = clean_text_batch(answers, nlp)

    for i, pair in enumerate(qa_pairs):
        pair["cleaned_question"] = cleaned_questions[i]
        pair["cleaned_answer"] = cleaned_answers[i]
    return qa_pairs

# Convertir une phrase en embedding avec Word2Vec
def sentence_to_embedding_word2vec(sentence, model):
    words = word_tokenize(sentence)
    embeddings = [model.wv[word] for word in words if word in model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

# Calcul de similarité cosinus
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Trouver la question la plus similaire
def find_most_similar_question(user_question, qa_pairs, model):
    user_vector = sentence_to_embedding_word2vec(user_question, model)
    similarities = [
        cosine_similarity(user_vector, pair["vector"]) for pair in qa_pairs
    ]
    best_match_index = similarities.index(max(similarities))
    return qa_pairs[best_match_index], max(similarities)

# Chargement des ressources
st.title("Chatbot basé sur Word2Vec")
st.write("### Initialisation...")

nlp = load_nlp()
qa_pairs = load_data()

# Charger ou nettoyer les données
if CACHE_FILE.exists():
    st.write("Chargement des données nettoyées depuis le fichier cache...")
    qa_pairs = load_cleaned_data()
else:
    st.write("Nettoyage des données, cela peut prendre du temps...")
    qa_pairs = preprocess_qa_pairs(qa_pairs, nlp)
    save_cleaned_data(qa_pairs)

# Charger ou entraîner le modèle Word2Vec
if MODEL_FILE.exists():
    st.write("Chargement du modèle Word2Vec depuis le fichier cache...")
    word2vec_model = load_model()
else:
    st.write("Entraînement du modèle Word2Vec...")
    corpus = [word_tokenize(pair["cleaned_question"]) for pair in qa_pairs]
    corpus += [word_tokenize(pair["cleaned_answer"]) for pair in qa_pairs]
    word2vec_model = Word2Vec(
        sentences=corpus,
        vector_size=300,
        window=5,
        min_count=2,
        workers=4,
    )
    save_model(word2vec_model)

# Calculer les vecteurs pour chaque question (une seule fois)
if "qa_vectors" not in st.session_state:
    st.write("Calcul des vecteurs pour les questions...")
    for pair in qa_pairs:
        pair["vector"] = sentence_to_embedding_word2vec(pair["cleaned_question"], word2vec_model)
    st.session_state.qa_vectors = qa_pairs
else:
    qa_pairs = st.session_state.qa_vectors

# Gestion de l'état de la discussion
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zone d'affichage de la discussion
st.write("### Discussion")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**Vous :** {message['content']}")
    else:
        st.markdown(f"**Chatbot :** {message['content']}")

# Entrée utilisateur
user_question = st.text_input("Posez votre question :")
if st.button("Envoyer") and user_question:
    # Ajouter la question de l'utilisateur à la discussion
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Trouver la meilleure réponse
    best_match, similarity = find_most_similar_question(user_question, qa_pairs, word2vec_model)
    response = f"{best_match['answer']} (similarité : {similarity:.2f})"

    # Ajouter la réponse du chatbot à la discussion
    st.session_state.messages.append({"role": "bot", "content": response})

    # Réactualiser pour afficher le message
    st.rerun()
