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
campaign = st.number_input("Número de contactos realizados durante la campaña actual:")
pdays = st.number_input("Días desde el último contacto con el cliente (si aplica):", min_value=-1, step=1)
previous = st.number_input("Número de contactos previos exitosos:")


if st.button("Predecir"):
   
    input_data = np.array([[age, balance, campaign, pdays, previous]])
    input_data_scaled = scaler.transform(input_data)
   
    prediction = model.predict(input_data_scaled)
    resultado = "Aceptará el producto" if prediction[0] == 1 else "No aceptará el producto"
    
    st.subheader(f"Resultado de la predicción: {resultado}")