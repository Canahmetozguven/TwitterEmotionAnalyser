import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import plotly
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from data_handler import DataCleaner, LabelHandler, DataMerger, DataImporter
st.title('Twitter Emotion Analysis')


@st.cache
def load_data():
    df_mutluluk = DataImporter(data_url="data/mutluluk_1000_df.csv").get_data()
    df_depresyon = DataImporter(data_url="data/depresyon_1000_df.csv").get_data()
    df_mutluluk = DataCleaner(data=df_mutluluk, label_columns="text", keyword="mutluluk").get_data()
    df_depresyon = DataCleaner(data=df_depresyon, label_columns="text", keyword="depresyon").get_data()
    df = DataMerger(data1=df_mutluluk, data2=df_depresyon).get_data()
    df.drop(columns="Unnamed: 0", inplace=True)
    return df
data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

pivot_t = pd.pivot_table(data, values="text", columns="label", aggfunc="count", index=data.datetime.dt.hour)
st.markdown('**When people send tweets about happiness or depression?**')
st.plotly_chart(px.bar(data_frame=pivot_t, x=pivot_t.index,
       y=["mutluluk","depresyon"],
       labels={
    "datetime": "Hour",
    "value": "Count"
                 }))
