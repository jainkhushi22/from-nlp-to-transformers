import re

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer


stop_words=set(stopwords.words("english"))

lemmatizer=WordNetLemmatizer()


def clean_text(text):

    text=re.sub(r'http\S+','',text)

    text=BeautifulSoup(text,'lxml').get_text()

    text=re.sub(r'[^a-zA-Z ]','',text)

    text=text.lower()

    text=" ".join([

        word for word in text.split()

        if word not in stop_words or word in ["not","no","never"]

    ])

    return text



def lemmatize_text(text):

    text=" ".join([

        lemmatizer.lemmatize(word,'v')

        for word in text.split()

    ])

    return text