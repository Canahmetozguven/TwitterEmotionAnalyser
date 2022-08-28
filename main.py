import streamlit as st
import pandas as pd
import plotly.express as px
from model.scraper import TwitterUserScraper
from model.model import Predictor
import wordcloud
from word_cloud.wordcloudmaker import WorldCloudMaker
import matplotlib.pyplot as plt
import time

st.title('Twitter Emotion Analysis')
user_name = str(st.text_input('Enter a Twitter user name:', placeholder="Enter a Twitter user name")).lower()

if user_name:
    if user_name.startswith('@'):
        user_name = user_name[1:]


    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def Data_Loading():
        data = TwitterUserScraper(user_name).get_tweets_df()
        data["pred"] = data[["text"]].apply(lambda x: Predictor(x["text"]).get_prediction(), axis=1)
        data["label"] = data["pred"].apply(lambda x: "Positive" if x == 1 else "Negative")
        return data


    data_load_state = st.text('Loading data...')
    df = Data_Loading()
    time.sleep(1)
    if df["text"].count() == 0:
        data_load_state.error('User not found please enter a valid user name')
        st.stop()
    data_load_state.text("Magic is done!")

    if df.pred.mean() < 0.5:
        st.write("The user is not feeling happy.")
    else:
        st.write("The user is feeling happy.")
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.dataframe(df.drop("quoted_tweet", axis=1))
    # tweet graph
    val = df.label.value_counts()
    st.markdown('**This is how much you have tweets sad and happy.**')
    st.plotly_chart(px.bar(data_frame=val, x=val.index, y=val.values,
                           labels={'index': 'Label', 'y': 'Count'}))

    # Word cloud
    st.markdown("**This is a word-cloud from your tweets!**")
    # Displaying word cloud
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.imshow(WorldCloudMaker(df).get_wordcloud(), interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot()

    # Tweets/hours graph
    st.markdown('**This is a graph of tweets/hours.**')
    pivot_t = pd.pivot_table(df, values="text", columns=["label"], aggfunc="count", index=df.datetime.dt.hour)
    tweets_hour = px.bar(data_frame=pivot_t, x=pivot_t.index,
                         y=["positive", "negative"],
                         labels={
                             "datetime": "Hour",
                             "value": "Count",
                             "variable": "Emotion",
                         })
    st.plotly_chart(tweets_hour, use_container_width=True)
