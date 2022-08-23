import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import nltk


nltk.download('stopwords')
STOPWORDS = stopwords.words('turkish')



class TextPreprocessor:
    def __init__(self, data, n_words=1000):
        self.data = data
        self.n_words = n_words
        self.preprocess_data()

    def preprocess_data(self):
        self.data["text"] = self.data["text"].str.lower()
        self.data['text'] = self.data['text'].str.replace(r'[^\w\s]+', '')
        self.data["text"] = self.data["text"].str.replace(r'\d', '', regex=True)
        self.data["text"] = self.data["text"].apply(
            lambda words: ' '.join(word.lower() for word in str(words).split() if word not in STOPWORDS))
        return self.data

    def get_data(self):
        return self.data
