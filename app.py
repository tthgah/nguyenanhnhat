import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

st.title("Nike Shoe Sales Clustering App")

uploaded_file = st.file_uploader("Upload Nike Shoe Sales CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Data Preview")
    st.write(df.head())

    st.subheader("📈 Data Info")
    buffer = []
    st.text(df.info())

    # Simple preprocessing: encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Standardize
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    st.subheader("🔗 Dendrogram")
    linked = linkage(df_scaled, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linked)
    st.pyplot(plt)

    st.subheader("🤖 Agglomerative Clustering")
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(df_scaled)
    df['Cluster'] = labels

    st.write(df[['Cluster']].value_counts().rename("Count"))
    st.write(df.head())
else:
    st.info("Please upload a CSV file to get started.")
