import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(8, activation='relu')(encoded)
    
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(decoded)
    output = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, output)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer=Adam(0.001), loss='mse')
    
    return autoencoder, encoder

def train_model(X_scaled):
    autoencoder, encoder = build_autoencoder(X_scaled.shape[1])
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, verbose=0)
    encoded_data = encoder.predict(X_scaled)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(encoded_data)
    
    return clusters
