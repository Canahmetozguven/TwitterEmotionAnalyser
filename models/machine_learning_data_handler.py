import pandas as pd
from nltk.corpus import stopwords
import numpy as np
STOPWORDS = stopwords.words('turkish')


class TextPreprocessor:
    def __init__(self, data):
        self.data = data
        self.preprocess_data()
        self.less_freq_word_deleter()

    def preprocess_data(self):
        self.data["text"] = self.data["text"].str.lower()
        self.data['text'] = self.data['text'].str.replace(r'[^\w\s]+', '')
        self.data["text"] = self.data["text"].str.replace(r'\d', '', regex=True)
        self.data["text"] = self.data["text"].apply(
            lambda words: ' '.join(word.lower() for word in str(words).split() if word not in STOPWORDS))
        return self.data

    def less_freq_word_deleter(self):
        """Delete words that occur less than threshold times"""
        all_ = [x for y in self.data["text"] for x in y.split(' ')]
        a, b = np.unique(all_, return_counts=True)
        to_remove = a[b < 3]
        self.data["text"] = [' '.join(np.array(y.split(' '))[~np.isin(y.split(' '), to_remove)]) for y in self.data["text"]]
        return self.data

    def get_data(self):
        return self.data
