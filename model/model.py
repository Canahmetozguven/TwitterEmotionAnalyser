import pandas as pd
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from copy import copy
from data_handler import DataImporter, DataCleaner, DataMerger
from model.machine_learning_data_handler import TextPreprocessor
pd.read_csv("data/mutluluk_1000_df.csv")
# Data Set
df_mutluluk = DataImporter(data_url="data/mutluluk_1000_df.csv").get_data()
df_depresyon = DataImporter(data_url="data/depresyon_1000_df.csv").get_data()
df_mutluluk = DataCleaner(data=df_mutluluk, label_columns="text", keyword="mutluluk").get_data()
df_depresyon = DataCleaner(data=df_depresyon, label_columns="text", keyword="depresyon").get_data()
df = DataMerger(data1=df_mutluluk, data2=df_depresyon).get_data()
df.label = LabelEncoder().fit_transform(df.label)
vectorizer = CountVectorizer()
df = TextPreprocessor(data=df).get_data()

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
X = vectorizer.fit_transform(df.text)
y = df.label
training_features, testing_features, training_target, testing_target = \
    train_test_split(X, y, random_state=None)

# Average CV score on the training set was: 0.8232204049844236
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        SelectFwe(score_func=f_classif, alpha=0.032)
    ),
    BernoulliNB(alpha=1.0, fit_prior=True)
)

exported_pipeline.fit(X, y)
results = exported_pipeline.predict(X)


class Predictor:
    def __init__(self, text):
        self.model = self.model = make_pipeline(
            make_union(
                FunctionTransformer(copy),
                SelectFwe(score_func=f_classif, alpha=0.032)
            ),
            BernoulliNB(alpha=1.0, fit_prior=True)
        ).fit(X, y)
        self.text = vectorizer.transform([text])
        self.predicter()

    def predicter(self):
        return self.model.predict(self.text)

    def get_prediction(self):
        return self.predicter()
