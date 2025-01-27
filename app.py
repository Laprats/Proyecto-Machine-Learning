import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


model_features = model.feature_names_in_
scaler_features = scaler.feature_names_in_

st.title("Predicción de Aceptación de Producto Bancario")

st.write("""
Ingrese los datos del cliente para predecir si aceptará o no el producto.
""")


age = st.number_input("Edad del cliente:", min_value=18, max_value=100, step=1)
balance = st.number_input("Balance promedio en la cuenta bancaria:")
campaign = st.number_input("Número de contactos realizados durante esta campaña:")
pdays = st.number_input("Número de días desde el último contacto previo:")
previous = st.number_input("Número de contactos previos realizados:")
housing = st.selectbox("¿Tiene hipoteca?", options=['Sí', 'No'])
loan = st.selectbox("¿Tiene préstamo personal?", options=['Sí', 'No'])


housing_map = {'No': 0, 'Sí': 1}
loan_map = {'No': 0, 'Sí': 1}


input_data = np.array([[age, balance, campaign, pdays, previous, housing_map[housing], loan_map[loan]]])
input_data_df = pd.DataFrame(input_data, columns=['age', 'balance', 'campaign', 'pdays', 'previous', 'housing', 'loan'])


for col in model_features:
    if col not in input_data_df.columns:
        input_data_df[col] = 0  


input_data_df = input_data_df[model_features]


st.write("Columnas en el DataFrame antes del escalador:", input_data_df.columns.tolist())

try:
    
    input_data_scaled = scaler.transform(input_data_df)
    
    
    st.write("Columnas reordenadas correctamente:")
    st.json(list(input_data_df.columns))

    
    if st.button("Predecir"):
        prediction = model.predict(input_data_scaled)
        resultado = "Aceptará el producto" if prediction[0] == 1 else "No aceptará el producto"
        st.subheader(f"Resultado de la predicción: {resultado}")

except Exception as e:
    st.error(f"Error al preparar los datos: {e}")

