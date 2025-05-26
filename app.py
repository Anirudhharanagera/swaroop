from model import train_model
from utils import load_and_preprocess_data

st.title("🧠 Market Segmentation using KMeans Clustering")

df, X_scaled = load_and_preprocess_data()
clusters, reduced_data = train_model(X_scaled)

df['Cluster'] = clusters

st.subheader("Clustered Data")
st.dataframe(df)

st.subheader("Cluster Counts")
st.bar_chart(df['Cluster'].value_counts())

# Optional: visualize PCA clustering
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
scatter = ax.scatter(reduced_data[:,0], reduced_data[:,1], c=clusters, cmap='viridis')
st.pyplot(fig)
