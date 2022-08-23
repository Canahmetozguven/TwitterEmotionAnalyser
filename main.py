import streamlit as st
import pandas as pd
import plotly.express as px
from model.scraper import TwitterUserScraper
from model.model import Predictor
import wordcloud
from word_cloud.wordcloudmaker import WorldCloudMaker
import matplotlib.pyplot as plt

st.title('Twitter Emotion Analysis')
user_name = str(st.text_input('Enter a Twitter user name:', placeholder="Enter a Twitter user name")).lower()
if user_name.startswith('@'):
    user_name = user_name[1:]
elif user_name:
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def Data_Loading():
        df = TwitterUserScraper(user_name).get_tweets_df()
        df["pred"] = df[["text"]].apply(lambda x: Predictor(x["text"]).get_prediction(), axis=1)
        df["label"] = df["pred"].apply(lambda x: "positive" if x == 1 else "negative")
        return df


    data_load_state = st.text('Loading data...')
    df = Data_Loading()
    data_load_state.text("Magic is done!")

    if df.pred.mean() < 0.5:
        st.write("The user is not feeling happy.")
    else:
        st.write("The user is feeling happy.")
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.dataframe(df.drop("quoted_tweet", axis=1))


    val = df.label.value_counts()
    st.markdown('**This is how much you have tweets sad and happy.**')
    st.plotly_chart(px.bar(data_frame=val, x=val.index, y=val.values,
                               labels={'index': 'Label', 'y': 'Count'}))

    st.markdown("**This is a word-cloud from your tweets!**")
    # Displaying word cloud
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.imshow(WorldCloudMaker(df).get_wordcloud(), interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot()
