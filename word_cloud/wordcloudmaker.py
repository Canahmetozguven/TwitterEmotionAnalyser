import numpy as np
from PIL import Image
from wordcloud import WordCloud
from os import path
import os
from model.machine_learning_data_handler import TextPreprocessor

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

class WorldCloudMaker:
    def __init__(self, data):
        self.data = TextPreprocessor(data).get_data()
        self.mask = np.array(Image.open(path.join(d, "logo.png")))
        self.make_wordcloud()
        self.text = None

    def make_wordcloud(self):
        self.text = " ".join(review for review in self.data["text"])
        wordcloud = WordCloud(background_color="white", max_words=2000, mask=self.mask, contour_width=2,
                              contour_color='steelblue').generate(self.text)
        return wordcloud

    def get_wordcloud(self):
        return self.make_wordcloud()
