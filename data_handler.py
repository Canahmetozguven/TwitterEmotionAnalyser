import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataImporter:
    def __init__(self, data_url):
        self.data_url = data_url
        self.data = None
        self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.data_url, parse_dates=['datetime'])

    def get_data(self):
        return self.data


class DataCleaner:
    def __init__(self, data, label_columns, keyword):
        """label_columns: which columns to extract label from
        keyword: keyword to extract label from"""
        self.data = data
        self.keyword = keyword
        self.label_columns = label_columns
        self.clean_data()
        self.label_extraction()

    def clean_data(self):
        """Drop unnecessary columns"""
        self.data = self.data.drop_duplicates()
        return self.data

    def label_extraction(self):
        """Extract label from data"""
        self.data[f"{self.label_columns}"] = self.data[f"{self.label_columns}"].str.replace(fr'{self.keyword}', '', regex=True)
        self.data["label"] = self.keyword
        return self.data

    def get_data(self):
        """Return cleaned data"""
        return self.data


class DataMerger:
    def __init__(self, data1, data2):
        """Merge two dataframes"""
        self.data = None
        self.data1 = data1
        self.data2 = data2
        self.merge_data()

    def merge_data(self):
        """Merge two dataframes"""
        self.data = pd.concat([self.data1, self.data2])
        self.data.drop(self.data[self.data["datetime"]=="bursacom16"].index, inplace=True)
        self.data["datetime"] = pd.to_datetime(self.data['datetime'], utc=True)
        return self.data

    def get_data(self):
        """Return merged data"""
        return self.data
