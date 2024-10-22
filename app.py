import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load data
vacantes = pd.read_csv('data/vacantes.csv')
competencias = pd.read_csv('data/competencias.csv')

# Load recommendation model
with open('models/recommendation_model.pkl', 'rb') as file:
    recommendation_model = pickle.load(file)

# Streamlit app
st.title("Inclusión Laboral de Personas con Discapacidad")

# User registration and profile creation
st.header("Registro y Creación de Perfil")
nombre = st.text_input("Nombre Completo")
email = st.text_input("Correo Electrónico")
discapacidad = st.selectbox("Tipo de Discapacidad", ["Visual", "Auditiva", "Motriz", "Intelectual", "Otra"])
habilidades = st.text_area("Habilidades y Competencias")

# Process input data
user_profile = {
    'nombre': nombre,
    'email': email,
    'discapacidad': discapacidad,
    'habilidades': habilidades.split(',')
}

# Show user profile
st.subheader("Perfil del Usuario")
st.write(user_profile)

# AI Recommendations
st.header("Recomendaciones de Vacantes")

if st.button("Obtener Recomendaciones"):
    # Dummy recommendation logic for illustration
    # Replace with the actual recommendation logic using the loaded model
    sample_vacantes = vacantes.sample(5)  # Randomly select 5 job postings for now
    st.write(sample_vacantes[['NOMBREAVISO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'ESPCD']])

# Add more features as needed

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Combine datasets for training
data = vacantes.merge(competencias, left_on='ID', right_on='AVISOID')

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['NOMBREAVISO'] + ' ' + data['NOMBRECOMPETENCIA'])

# Model training
model = NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(X)

# Save the model
with open('models/recommendation_model.pkl', 'wb') as file:
    pickle.dump(model, file)