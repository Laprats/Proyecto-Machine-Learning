import streamlit as st
import pickle
import numpy as np


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.title("Predicción de Aceptación de Producto Bancario")


st.write("""
Ingrese los datos del cliente para predecir si aceptará o no el producto.
""")


age = st.number_input("Edad del cliente:", min_value=18, max_value=100, step=1)
balance = st.number_input("Balance promedio en la cuenta bancaria:")
education = st.selectbox("Nivel educativo del cliente:", options=['Primario', 'Secundario', 'Terciario'])
housing = st.selectbox("¿Tiene hipoteca?", options=['Sí', 'No'])
loan = st.selectbox("¿Tiene préstamo personal?", options=['Sí', 'No'])


education_map = {'Primario': 0, 'Secundario': 1, 'Terciario': 2}
housing_map = {'No': 0, 'Sí': 1}
loan_map = {'No': 0, 'Sí': 1}



input_data = np.array([[age, balance, education_map[education], housing_map[housing], loan_map[loan]]])
input_data_scaled = scaler.transform(input_data)


if st.button("Predecir"):
    prediction = model.predict(input_data_scaled)
    resultado = "Aceptará el producto" if prediction[0] == 1 else "No aceptará el producto"
    st.subheader(f"Resultado de la predicción: {resultado}")
