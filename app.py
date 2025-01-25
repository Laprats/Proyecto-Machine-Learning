import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


model_features = list(model.feature_names_in_)
scaler_features = list(scaler.feature_names_in_)

st.title("Predicción de Aceptación de Producto Bancario")


age = st.number_input("Edad del cliente:", min_value=18, max_value=100, step=1)
balance = st.number_input("Balance promedio en la cuenta bancaria:")
campaign = st.number_input("Número de campañas previas:")
pdays = st.number_input("Días previos al contacto anterior:")
previous = st.number_input("Número de contactos previos:")
housing = st.selectbox("¿Tiene hipoteca?", options=["Sí", "No"])
loan = st.selectbox("¿Tiene préstamo personal?", options=["Sí", "No"])


housing_map = {"No": 0, "Sí": 1}
loan_map = {"No": 0, "Sí": 1}


input_data = pd.DataFrame([{
    "age": age,
    "balance": balance,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "housing": housing_map[housing],
    "loan": loan_map[loan]
}])


for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0  
input_data = input_data[model_features]

st.write("Columnas en el DataFrame antes del escalador:", input_data.columns.tolist())


try:
    input_data_scaled = scaler.transform(input_data)
    st.write("Datos escalados:", input_data_scaled)
except Exception as e:
    st.error(f"Error al escalar los datos: {e}")
    input_data_scaled = None


if st.button("Predecir"):
    if input_data_scaled is not None:
        try:
            prediction = model.predict(input_data_scaled)
            resultado = "Aceptará el producto" if prediction[0] == 1 else "No aceptará el producto"
            st.subheader(f"Resultado de la predicción: {resultado}")
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
    else:
        st.error("No se pudieron procesar los datos de entrada.")
