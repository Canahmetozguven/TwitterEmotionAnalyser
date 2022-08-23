import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from data_handler import DataImporter, DataCleaner, DataMerger
from model.machine_learning_data_handler import TextPreprocessor
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import nltk

nltk.download('punkt')


def multiple_word_remove_func(text, words_2_remove_list):
    """
    Removes certain words from string, if present

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Removes the defined words from the created tokens

    Args:
        text (str): String to which the functions are to be applied, string
        words_2_remove_list (list): Words to be removed from the text, list of strings

    Returns:
        String with removed words
    """
    words_to_remove_list = words_2_remove_list

    words = word_tokenize(text)
    text = ' '.join([word for word in words if word not in words_to_remove_list])
    return text


def most_rare_word_func(text, n_words=5):
    """
    Returns the most rarely used words from a text

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to determine the word frequency

    Args:
        text (str): String to which the functions are to be applied, string

    Returns:
        List of the most rarely occurring words (by default = 5)
        :param text:
        :param n_words:
    """
    words = word_tokenize(text)
    fdist = FreqDist(words)

    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=False)

    n_words = n_words
    most_rare_words_list = list(df_fdist['Word'][-n_words:])

    return most_rare_words_list


# Data Set
df_mutluluk = DataImporter(data_url="data/mutluluk.csv").get_data()
df_depresyon = DataImporter(data_url="data/depresyon.csv").get_data()
df_mutluluk = DataCleaner(data=df_mutluluk, label_columns="text", keyword="mutluluk").get_data()
df_depresyon = DataCleaner(data=df_depresyon, label_columns="text", keyword="depresyon").get_data()
df = DataMerger(data1=df_mutluluk, data2=df_depresyon).get_data()
df.label = LabelEncoder().fit_transform(df.label)
vectorizer = CountVectorizer()
df = TextPreprocessor(data=df, n_words=1000).get_data()
text_corpus_original = df['text'].str.cat(sep=' ')
most_rare_words_list_DataFrame = most_rare_word_func(text_corpus_original, n_words=1000)
df["text"] = df.apply(lambda x: multiple_word_remove_func(x["text"],
                                                          most_rare_words_list_DataFrame), axis=1)

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
X = vectorizer.fit_transform(df.text)
y = df.label
training_features, testing_features, training_target, testing_target = \
    train_test_split(X, y, random_state=None)

# Average CV score on the training set was: 0.87
exported_pipeline = MultinomialNB(alpha=1.0, fit_prior=False)
exported_pipeline.fit(training_features, training_target)


class Predictor:
    def __init__(self, text):
        self.model = MultinomialNB(alpha=1.0, fit_prior=False).fit(training_features, training_target)
        self.text = vectorizer.transform([text])
        self.predicter()

    def predicter(self):
        return self.model.predict(self.text)

    def get_prediction(self):
        return self.predicter()
