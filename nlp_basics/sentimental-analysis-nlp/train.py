import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import clean_text
from src.preprocess import lemmatize_text
from src.embeddings import get_document_vector
from src.ml_core import build_model
from src.ml_core import evaluate_model

data = pd.read_csv("kindle_reviews.csv",nrows=12000)

# Select required columns
data = data[['reviewText','overall']]

# Remove missing values
data.dropna(inplace=True)


# Convert ratings to sentiment
# 0 = negative (1,2,3)
# 1 = positive (4,5)

data['overall'] = data['overall'].apply(

    lambda x: 0 if x < 4 else 1

)


# Define target
y = data['overall']



print("Preprocessing text...")

data['reviewText'] = data['reviewText'].apply(clean_text)

data['reviewText'] = data['reviewText'].apply(lemmatize_text)


X = data['reviewText']

y = data['overall']


print("Splitting dataset...")

X_train_text,X_test_text,y_train,y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


##################################################
# Bag of Words Model
##################################################

print("\nTraining BoW model...")

bow = CountVectorizer(max_features=5000)

X_train_bow = bow.fit_transform(X_train_text)

X_test_bow = bow.transform(X_test_text)

bow_model = build_model()

bow_model.fit(X_train_bow,y_train)

bow_acc,_,_,bow_f1 = evaluate_model(

    bow_model,
    X_test_bow,
    y_test

)

print("BoW Accuracy:",bow_acc)


##################################################
# TFIDF Model
##################################################

print("\nTraining TFIDF model...")

tfidf = TfidfVectorizer(

    max_features=5000,
    ngram_range=(1,2)

)

X_train_tfidf = tfidf.fit_transform(X_train_text)

X_test_tfidf = tfidf.transform(X_test_text)

tfidf_model = build_model()

tfidf_model.fit(X_train_tfidf,y_train)

tfidf_acc,_,_,tfidf_f1 = evaluate_model(

    tfidf_model,
    X_test_tfidf,
    y_test

)

print("TFIDF Accuracy:",tfidf_acc)


##################################################
# Word2Vec Model
##################################################

print("\nTraining Word2Vec model...")

X_train_vec = X_train_text.apply(get_document_vector)

X_test_vec = X_test_text.apply(get_document_vector)

X_train_vec = list(X_train_vec)

X_test_vec = list(X_test_vec)

w2v_model = build_model()

w2v_model.fit(X_train_vec,y_train)

w2v_acc,report,matrix,w2v_f1 = evaluate_model(

    w2v_model,
    X_test_vec,
    y_test

)

print("Word2Vec Accuracy:",w2v_acc)


##################################################
# Model comparison
##################################################

results = pd.DataFrame({

    "Model":[

        "Bag of Words",
        "TFIDF",
        "Word2Vec"

    ],

    "Accuracy":[

        bow_acc,
        tfidf_acc,
        w2v_acc

    ],

    "F1 Score":[

        bow_f1,
        tfidf_f1,
        w2v_f1

    ]

})


print("\nModel Comparison:")

print(results)


##################################################
# Save artifacts
##################################################

if not os.path.exists("artifacts"):

    os.makedirs("artifacts")

results.to_csv(

    "artifacts/model_results.csv",
    index=False

)


with open(

    "artifacts/metrics.txt",
    "w"

) as f:

    f.write(report)


##################################################
# Save model (Word2Vec here)
##################################################
if not os.path.exists("models"):

    os.makedirs("models")

pickle.dump(
    bow_model,
    open("models/bow_model.pkl","wb")
)

pickle.dump(
    tfidf_model,
    open("models/tfidf_model.pkl","wb")
)

pickle.dump(
    w2v_model,
    open("models/sentiment_model.pkl","wb")
)


# Save vectorizers
pickle.dump(
    bow,
    open("models/bow_vectorizer.pkl","wb")
)

pickle.dump(
    tfidf,
    open("models/tfidf_vectorizer.pkl","wb")
)


print("All models saved")