import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    df = pd.read_csv("sample_data.csv")
    features = df.drop(columns=['CustomerID'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return df, X_scaled
