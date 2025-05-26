import streamlit as st
import pandas as pd
from model import train_model
from utils import load_and_preprocess_data

st.title("🧠 Advanced Market Segmentation using Deep Clustering")

df, X_scaled = load_and_preprocess_data()
clusters = train_model(X_scaled)
df['Cluster'] = clusters

st.subheader("Clustered Data")
st.dataframe(df)

st.subheader("Cluster Summary")
st.bar_chart(df['Cluster'].value_counts())
