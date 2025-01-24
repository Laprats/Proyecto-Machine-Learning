import streamlit as st
import pickle
import numpy as np
import pandas as pd


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
pdays = st.number_input("Número de días desde el último contacto:", min_value=-1, step=1)
previous = st.number_input("Número de contactos previos en la campaña:", min_value=0, step=1)
education = st.selectbox("Nivel educativo del cliente:", options=['Primario', 'Secundario', 'Terciario'])
housing = st.selectbox("¿Tiene hipoteca?", options=['Sí', 'No'])
loan = st.selectbox("¿Tiene préstamo personal?", options=['Sí', 'No'])


education_map = {'Primario': 0, 'Secundario': 1, 'Terciario': 2}
housing_map = {'No': 0, 'Sí': 1}
loan_map = {'No': 0, 'Sí': 1}


input_data = np.array([[age, balance, pdays, previous, education_map[education], housing_map[housing], loan_map[loan]]])
input_data_df = pd.DataFrame(input_data, columns=['age', 'balance', 'pdays', 'previous', 'education', 'housing', 'loan'])


try:
    
    input_data_df['balance_to_age_ratio'] = input_data_df['balance'] / (input_data_df['age'] + 1)

    bins = [-float('inf'), -1, 0, 1, float('inf')]
    labels = ['muy bajo', 'bajo', 'medio', 'alto']
    input_data_df['balance_category'] = pd.cut(input_data_df['balance'], bins=bins, labels=labels)

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    input_data_df['balance_category'] = encoder.fit_transform(input_data_df['balance_category'])

    input_data_df['cluster'] = 2  

    
    expected_features = list(model.feature_names_in_)
    input_data_df = input_data_df[expected_features]

    
    input_data_scaled = scaler.transform(input_data_df)

except Exception as e:
    st.error(f"Error al preparar los datos: {e}")
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
        st.error("No se pudieron preparar los datos. Verifica los valores ingresados.")
