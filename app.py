import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset and preprocess it
def load_and_preprocess_data():
    # Dataset for training
    data = pd.read_csv("Mall_Customers.csv")

    # Encode categorical data
    data['Genre'] = LabelEncoder().fit_transform(data['Genre'])
    
    # Features to be used for clustering
    features = data[['Age', 'Annual_Income_(k$)', 'Spending_Score']]
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return data, features_scaled, scaler

# Train the KMeans model
def train_kmeans(features_scaled):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features_scaled)
    return kmeans

# Define cluster descriptions
CLUSTER_LABELS = {
    0: "Low Income, Low Spending",
    1: "High Income, High Spending",
    2: "Moderate Income, Moderate Spending"
}

# Predict the cluster for a new customer
def predict_cluster(kmeans, scaler, age, annual_income, spending_score):
    # Prepare input
    user_data = np.array([[age, annual_income, spending_score]])
    user_data_scaled = scaler.transform(user_data)
    cluster = kmeans.predict(user_data_scaled)[0]
    return CLUSTER_LABELS[cluster]

# Streamlit application
def customer_segmentation_app():
    st.title("Customer Segmentation Using KMeans Clustering")
    
    # Training phase
    data, features_scaled, scaler = load_and_preprocess_data()
    kmeans = train_kmeans(features_scaled)
    
    
    # Prediction phase
    st.header("Predict Customer Cluster")
    st.write("Enter customer details to predict their cluster.")
    
    # Input fields
    genre = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    annual_income = st.number_input("Annual Income (in $1000)", min_value=0, max_value=500, step=1)
    spending_score = st.number_input("Spending Score (0-100)", min_value=0, max_value=100, step=1)
    
    if st.button("Predict Cluster"):
        # Encode gender
        genre_encoded = 0 if genre == 'Male' else 1
        # Predict cluster
        cluster = predict_cluster(kmeans, scaler, age, annual_income, spending_score)
        st.success(f"The customer belongs to Cluster {cluster}.")

# Run the application
if __name__ == "__main__":
    customer_segmentation_app()
