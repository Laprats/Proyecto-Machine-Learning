import streamlit as st
import pickle
import numpy as np
import pandas as pd


st.title("Predicción de Aceptación de Producto Bancario")
st.write("Ingrese los datos del cliente para predecir si aceptará o no el producto.")


age = st.number_input("Edad del cliente:", min_value=18, max_value=100, step=1)
balance = st.number_input("Balance promedio en la cuenta bancaria:")
campaign = st.number_input("Número de contactos realizados durante esta campaña:")
housing = st.selectbox("¿Tiene hipoteca?", options=['Sí', 'No'])
loan = st.selectbox("¿Tiene préstamo personal?", options=['Sí', 'No'])
balance_to_age_ratio = balance / age  # Relación entre balance y edad
balance_category = pd.cut([balance], bins=[-float('inf'), 0, 1000, 5000, float('inf')], labels=['Muy bajo', 'Bajo', 'Medio', 'Alto'])[0]


housing_map = {'No': 0, 'Sí': 1}
loan_map = {'No': 0, 'Sí': 1}
balance_category_map = {'Muy bajo': 0, 'Bajo': 1, 'Medio': 2, 'Alto': 3}


input_data = np.array([[housing_map[housing], loan_map[loan], balance_to_age_ratio, balance_category_map[balance_category]]])

try:
    input_data_df = pd.DataFrame(input_data, columns=['housing', 'loan', 'balance_to_age_ratio', 'balance_category'])

    
    st.write("Columnas procesadas antes de escalar:", list(input_data_df.columns))

    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        st.write("Columnas esperadas por el escalador:", scaler.feature_names_in_)

   
    input_data_scaled = scaler.transform(input_data_df)
except Exception as e:
    st.error(f"Error al preparar los datos: {e}")
    input_data_scaled = None


try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
        st.write("Columnas esperadas por el modelo:", model.feature_names_in_)

    if st.button("Predecir"):
        if input_data_scaled is not None:
            prediction = model.predict(input_data_scaled)
            resultado = "Aceptará el producto" if prediction[0] == 1 else "No aceptará el producto"
            st.subheader(f"Resultado de la predicción: {resultado}")
        else:
            st.error("No se pudieron preparar los datos. Verifica los valores ingresados.")
except Exception as e:
    st.error(f"Error al cargar el modelo o realizar la predicción: {e}")
