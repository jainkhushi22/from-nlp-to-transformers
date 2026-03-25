import streamlit as st
import pickle
import pandas as pd
import os

from src.preprocess import clean_text
from src.preprocess import lemmatize_text
from src.embeddings import get_document_vector


st.set_page_config(
    page_title="Sentiment Analysis",
    layout="centered"
)


st.title("Amazon Review Sentiment Analysis")

st.write("Predict sentiment using NLP models")


###################################################
# Check models exist
###################################################

if not os.path.exists("models/bow_model.pkl"):

    st.error("Models not found. Run train.py first")

    st.stop()


###################################################
# Load models
###################################################

bow_model = pickle.load(
open("models/bow_model.pkl","rb")
)

tfidf_model = pickle.load(
open("models/tfidf_model.pkl","rb")
)

w2v_model = pickle.load(
open("models/sentiment_model.pkl","rb")
)

bow_vectorizer = pickle.load(
open("models/bow_vectorizer.pkl","rb")
)

tfidf_vectorizer = pickle.load(
open("models/tfidf_vectorizer.pkl","rb")
)


###################################################
# Sidebar menu
###################################################

menu = st.sidebar.selectbox(

"Menu",

["Predict","Model Comparison","About"]

)


###################################################
# Prediction section
###################################################

if menu=="Predict":

    model_choice = st.selectbox(

    "Select Model",

    ["TFIDF","Bag of Words","Word2Vec"]

    )


    text = st.text_area(

    "Enter Review"

    )


    if st.button("Predict"):

        if len(text.strip()) < 3:

            st.warning("Enter a longer review")

            st.stop()


        text = clean_text(text)

        text = lemmatize_text(text)


        if model_choice=="Bag of Words":

            vec = bow_vectorizer.transform([text])

            pred = bow_model.predict(vec)

            prob = bow_model.predict_proba(vec)


        elif model_choice=="TFIDF":

            vec = tfidf_vectorizer.transform([text])

            pred = tfidf_model.predict(vec)

            prob = tfidf_model.predict_proba(vec)


        else:

            vec = get_document_vector(text)

            pred = w2v_model.predict([vec])

            prob = w2v_model.predict_proba([vec])


        if pred[0]==1:

            st.success("Positive Review")

        else:

            st.error("Negative Review")


        st.write(

        "Confidence:",

        round(max(prob[0]),3)

        )


###################################################
# Model comparison section
###################################################

if menu=="Model Comparison":

    st.subheader("Model Performance")

    results = pd.read_csv(

    "artifacts/model_results.csv"

    )

    st.table(results)


###################################################
# About section
###################################################

if menu=="About":

    st.write("""

### Models Used

TFIDF (Best performing)

Bag of Words

Word2Vec


### ML Model

Logistic Regression , naive bias(gaussian NB)


### Project Features

Text preprocessing

Feature engineering comparison

Binary sentiment classification

Model evaluation

Streamlit deployment

""")