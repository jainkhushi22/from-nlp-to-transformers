# Amazon Review Sentiment Analysis using NLP

## Project Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline to classify Amazon Kindle reviews as **Positive** or **Negative**. The project compares multiple feature engineering techniques including **Bag of Words (BoW)**, **TF-IDF**, and **Word2Vec** using Logistic Regression.

The best performing model is deployed using **Streamlit** for interactive sentiment prediction.

---

## Features

- Text preprocessing pipeline
- Stopword removal and lemmatization
- Feature engineering comparison:
  - Bag of Words
  - TF-IDF
  - Word2Vec
- Logistic Regression classification
- Model evaluation using Accuracy and F1 score
- Streamlit web application
- Model comparison dashboard

---

## Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Bag of Words | 80% | 0.80 |
| TFIDF | 82% | 0.82 |
| Word2Vec | 76% | 0.78 |

**Conclusion:**
TF-IDF performed best because it preserves word importance, while Word2Vec captures semantic similarity but loses sentiment polarity due to vector averaging.

---

## Project Structure
sentiment-analysis-nlp/

app.py # Streamlit application
train.py # Model training pipeline
requirements.txt # Dependencies

notebooks/
review.ipynb # Experimentation notebook

src/
preprocess.py # Text preprocessing
embeddings.py # Word2Vec embeddings
ml_core.py # Model utilities

artifacts/
model_results.csv # Model comparison results

models/ # Saved models (ignored)

---
## Dataset

This project uses Amazon Kindle review data for sentiment classification.

Features:

reviewText → input text  
overall → rating (used to derive sentiment)

Sentiment mapping:

1–3 → Negative  
4–5 → Positive  

Total samples used: 12,000

Data preprocessing included:

- Text cleaning
- HTML removal
- Stopword removal
- Lemmatization
- URL removal

Dataset not included in repository due to size constraints.


---

## Installation

Clone repository:
git clone https://github.com/jainkhushi22/from-nlp-to-transformers.git


Install dependencies:
pip install -r requirements.txt


---

## Training Models

Run:
python train.py

This will:
- Train BoW model
- Train TFIDF model
- Train Word2Vec model
- Save model comparison results

---

## Running the Application

Run Streamlit app:
streamlit run app.py


Enter a review and select model to predict sentiment.

---

## Example Predictions

Example inputs:
This product is amazing → Positive
Waste of money → Negative
Good quality and worth buying → Positive


---

## Technologies Used

- Python
- Scikit-learn
- NLTK
- Gensim
- Streamlit
- Pandas
- NumPy

---

## Future Improvements

- Transformer models (BERT)
- Hyperparameter tuning
- Cross validation
- Model monitoring
- Deployment on cloud

---

## Author

Khushi Jain
AI & Data Science Student
