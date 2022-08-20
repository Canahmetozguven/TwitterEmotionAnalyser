import streamlit as st
import pandas as pd
import plotly.express as px
from model.scraper import TwitterUserScraper
from model.model import Predictor

st.title('Twitter Emotion Analysis')
user_name = str(st.text_input('Enter a Twitter user name:', placeholder="Enter a Twitter user name"))

if user_name:

    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def chacher():
        df = TwitterUserScraper(user_name).get_tweets_df()
        df["pred"] = df[["text"]].apply(lambda x: Predictor(x["text"]).get_prediction(), axis=1)
        df["label"] = df["pred"].apply(lambda x: "positive" if x == 1 else "negative")
        return df

    data_load_state = st.text('Loading data...')
    df = chacher()
    data_load_state.text("Done! (using st.cache)")

    if df.pred.mean() < 0.5:
        st.write("The user is not feeling happy.")
    else:
        st.write("The user is feeling happy.")
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(df.drop("quoted_tweet", axis=1))

    val = df.label.value_counts()
    st.markdown('**When people send tweets about happiness or depression?**')
    st.plotly_chart(px.bar(data_frame=val, x=val.index, y=val.values,
                           labels={'index': 'Label', 'y': 'Count'}))
