import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


model_features = getattr(model, 'feature_names_in_', None)
scaler_features = getattr(scaler, 'feature_names_in_', None)

st.title("Predicción de Aceptación de Producto Bancario")

#
if model_features:
    st.write(f"Características esperadas por el modelo: {model_features}")
else:
    st.warning("El modelo no proporciona las características esperadas.")

if scaler_features:
    st.write(f"Características esperadas por el escalador: {scaler_features}")
else:
    st.warning("El escalador no proporciona las características esperadas.")


age = st.number_input("Edad del cliente:", min_value=18, max_value=100, step=1)
balance = st.number_input("Balance promedio en la cuenta bancaria:")
education = st.selectbox("Nivel educativo del cliente:", options=['Primario', 'Secundario', 'Terciario'])
housing = st.selectbox("¿Tiene hipoteca?", options=['Sí', 'No'])
loan = st.selectbox("¿Tiene préstamo personal?", options=['Sí', 'No'])


education_map = {'Primario': 0, 'Secundario': 1, 'Terciario': 2}
housing_map = {'No': 0, 'Sí': 1}
loan_map = {'No': 0, 'Sí': 1}


input_data = pd.DataFrame([{
    'age': age,
    'balance': balance,
    'education': education_map[education],
    'housing': housing_map[housing],
    'loan': loan_map[loan]
}])


if model_features and not set(input_data.columns).issubset(model_features):
    st.error(f"Las columnas proporcionadas no coinciden con las esperadas por el modelo: {model_features}")
else:
    try:
        s
        if scaler_features:
            input_data_scaled = scaler.transform(input_data[scaler_features])
        else:
            input_data_scaled = scaler.transform(input_data)

        
        if st.button("Predecir"):
            prediction = model.predict(input_data_scaled)
            resultado = "Aceptará el producto" if prediction[0] == 1 else "No aceptará el producto"
            st.subheader(f"Resultado de la predicción: {resultado}")
    except Exception as e:
        st.error(f"Error durante la escalación o predicción: {e}")

