import gensim.downloader as api

import numpy as np


wv=api.load("word2vec-google-news-300")


def get_document_vector(text):

    words=text.split()

    valid_words=[

        word for word in words

        if word in wv

    ]

    if len(valid_words)==0:

        return np.zeros(300)

    vectors=[

        wv[word]

        for word in valid_words

    ]

    return np.mean(vectors,axis=0)