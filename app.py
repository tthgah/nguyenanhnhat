import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Nike Sales Visualization", layout="wide")

st.title("📊 Nike Shoe Sales Clustering & Visualization")

# Dữ liệu mẫu tích hợp sẵn (giả lập từ file nike shoe sales data.csv)
data = {
    'Product': ['Air Max', 'Revolution', 'Pegasus', 'ZoomX', 'React'],
    'Category': ['Running', 'Running', 'Running', 'Training', 'Casual'],
    'Price': [120, 90, 100, 140, 110],
    'Units Sold': [500, 300, 400, 200, 250],
    'Revenue': [60000, 27000, 40000, 28000, 27500]
}
df = pd.DataFrame(data)

# Encode dữ liệu dạng object → số
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])
    
# Chuẩn hóa dữ liệu
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Áp dụng Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3)
df['Cluster'] = model.fit_predict(df_scaled)

# Biểu đồ 1: Revenue by Product
st.subheader("1️⃣ Revenue by Product")
fig1, ax1 = plt.subplots()
sns.barplot(x='Product', y='Revenue', data=df, ax=ax1)
st.pyplot(fig1)

# Biểu đồ 2: Revenue by Category
st.subheader("2️⃣ Revenue by Category")
fig2, ax2 = plt.subplots()
sns.barplot(x='Category', y='Revenue', data=df, ax=ax2)
st.pyplot(fig2)

# Biểu đồ 3: Price vs Units Sold (có phân cụm)
st.subheader("3️⃣ Price vs Units Sold")
fig3, ax3 = plt.subplots()
sns.scatterplot(x='Price', y='Units Sold', hue='Cluster', data=df, ax=ax3)
st.pyplot(fig3)

# Biểu đồ 4: Heatmap tương quan
st.subheader("4️⃣ Correlation Heatmap")
fig4, ax4 = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

# Biểu đồ 5: Dendrogram
st.subheader("5️⃣ Dendrogram")
linked = linkage(df_scaled, method='ward')
fig5, ax5 = plt.subplots(figsize=(10, 5))
dendrogram(linked, ax=ax5)
st.pyplot(fig5)
