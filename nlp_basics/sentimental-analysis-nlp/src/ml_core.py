from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from src.preprocess import clean_text
from src.preprocess import lemmatize_text

from src.embeddings import get_document_vector


# Build model

def build_model():

    model=LogisticRegression(

        max_iter=500,

        class_weight='balanced',

        random_state=42

    )

    return model


# Evaluate model

def evaluate_model(model,X_test,y_test):

    pred=model.predict(X_test)

    accuracy=accuracy_score(y_test,pred)

    report=classification_report(y_test,pred)

    matrix=confusion_matrix(y_test,pred)

    f1=f1_score(

        y_test,

        pred,

        average='weighted'

    )

    return accuracy,report,matrix,f1


# Prediction pipeline

def predict(text,model):

    text=clean_text(text)

    text=lemmatize_text(text)

    vector=get_document_vector(text)

    pred=model.predict([vector])

    prob=model.predict_proba([vector])

    return pred[0],max(prob[0])