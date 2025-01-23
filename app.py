{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bd4d0de2-c54d-4886-8fae-59b08c637c95",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "866a2acf-7ab3-4783-8f6d-b08ee5250a08",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('model.pkl', 'rb') as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "\n",
    "with open('scaler.pkl', 'rb') as file:\n",
    "    scaler = pickle.load(file)\n",
    "\n",
    "\n",
    "st.title(\"Predicción de Aceptación de Producto Bancario\")\n",
    "\n",
    "st.write(\"\"\"\n",
    "Ingrese los datos del cliente para predecir si aceptará o no el producto.\n",
    "\"\"\")\n",
    "\n",
    "\n",
    "age = st.number_input(\"Edad del cliente:\", min_value=18, max_value=100, step=1)\n",
    "balance = st.number_input(\"Balance promedio en la cuenta bancaria:\")\n",
    "education = st.selectbox(\"Nivel educativo del cliente:\", options=['Primario', 'Secundario', 'Terciario'])\n",
    "housing = st.selectbox(\"¿Tiene hipoteca?\", options=['Sí', 'No'])\n",
    "loan = st.selectbox(\"¿Tiene préstamo personal?\", options=['Sí', 'No'])\n",
    "\n",
    "\n",
    "education_map = {'Primario': 0, 'Secundario': 1, 'Terciario': 2}\n",
    "housing_map = {'No': 0, 'Sí': 1}\n",
    "loan_map = {'No': 0, 'Sí': 1}\n",
    "\n",
    "\n",
    "input_data = np.array([[age, balance, education_map[education], housing_map[housing], loan_map[loan]]])\n",
    "\n",
    "\n",
    "try:\n",
    "    input_data_df = pd.DataFrame(input_data, columns=['age', 'balance', 'education', 'housing', 'loan'])\n",
    "\n",
    "    \n",
    "    input_data_scaled = scaler.transform(input_data_df)\n",
    "except Exception as e:\n",
    "    st.error(f\"Error al preparar los datos: {e}\")\n",
    "    input_data_scaled = None\n",
    "\n",
    "\n",
    "if st.button(\"Predecir\"):\n",
    "    if input_data_scaled is not None:\n",
    "        try:\n",
    "            prediction = model.predict(input_data_scaled)\n",
    "            resultado = \"Aceptará el producto\" if prediction[0] == 1 else \"No aceptará el producto\"\n",
    "            st.subheader(f\"Resultado de la predicción: {resultado}\")\n",
    "        except Exception as e:\n",
    "            st.error(f\"Error durante la predicción: {e}\")\n",
    "    else:\n",
    "        st.error(\"No se pudieron escalar los datos de entrada. Verifica los valores proporcionados.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
