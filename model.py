import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def train_model(X_scaled, n_clusters=4):
    # Optional: reduce dimensions for cleaner clustering
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X_scaled)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    return clusters, reduced_data
