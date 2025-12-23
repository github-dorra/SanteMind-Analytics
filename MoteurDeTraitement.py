import pandas as pd
import numpy as np 
from textblob import TextBlob
import spacy
import re

# Chargement du modèle Spacy
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """Nettoyage du texte brut (Phase NLP)"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def get_final_features(user_input, model_pack):
    
    # 1. Nettoyage
    cleaned_text = preprocess_text(user_input)
    
    # 2. TF-IDF 
    input_tfidf = model_pack['tfidf'].transform([cleaned_text]).toarray()
    
    # 3. Extraction 
    blob = TextBlob(user_input)
    pol = blob.sentiment.polarity
    sub = blob.sentiment.subjectivity
    
    
    # On peut mettre 0 ou chercher des mots clés spécifiques*
    t=user_input.lower()
    
    lex_liwc_anx = 40.0 if any(w in t for w in ['anxity', 'panic', 'fear', 'scare', 'worr', 'terrif']) else 0.0
    lex_liwc_anger = 40.0 if any(w in t for w in ['angry', 'hate', 'mad', 'furi', 'annoyone']) else 0.0        
    lex_liwc_sad = 40.0 if any(w in t for w in ['sad', 'depress', 'cry', 'alone', 'hopeless', 'break']) else 0.0
    lex_liwc_health = 40.0 if any(w in t for w in ['pain', 'weak', 'tired','sick', 'hurt', 'docter', 'hospital', 'headache']) else 0.0
    lex_liwc_work = 40.0 if any(w in t for w in ['work', 'job', 'boss', 'deadline', 'office', 'career']) else 0.0
    lex_liwc_family = 40.0 if any(w in t for w in ['family', 'mom', 'dad', 'parent', 'brother', 'sister']) else 0.0
    lex_liwc_friend = 40.0 if any(w in t for w in ['friend', 'colleague', 'people', 'buddy']) else 0.0
    lex_liwc_i = 40.0 if 'i ' in t or "i'm" in t or "my " in t else 0.0

        # Recalcul des index (Identique au Notebook)
    distress_index = lex_liwc_health + lex_liwc_sad + lex_liwc_i
    burnout_index = lex_liwc_work + lex_liwc_anx+ lex_liwc_anger

    
    # Liste pour SVM (13 colonnes numériques)
    svm_numeric_list = [
       1.77, lex_liwc_anx, lex_liwc_anger,
        lex_liwc_sad, lex_liwc_health, lex_liwc_work,
        lex_liwc_family, lex_liwc_friend, lex_liwc_i,
        pol, sub, distress_index, burnout_index
    ]
    numeric_feats = np.array(svm_numeric_list).reshape(1, -1)
    
    cleaned_text = preprocess_text(user_input)
    input_tfidf = model_pack['tfidf'].transform([cleaned_text]).toarray()
    X_lda = model_pack['lda_model'].transform(input_tfidf)
    
    block_to_scale = np.hstack([numeric_feats, X_lda])

    if 'scaler_svm' in model_pack:
        block_scaled = model_pack['scaler_svm'].transform(block_to_scale)
    else:
        block_scaled = block_to_scale
        
    X_final_svm = np.hstack([input_tfidf, block_scaled])

    kmeans_list = [distress_index, burnout_index, pol, sub, lex_liwc_anx, lex_liwc_anger, lex_liwc_sad, lex_liwc_work, lex_liwc_family, lex_liwc_health]
    X_km_scaled = model_pack['scaler'].transform([kmeans_list])
    
    return X_final_svm, X_km_scaled