{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "45a15dc8-a5b2-474c-aca2-812877dd44a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\aprats\\AppData\\Local\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "with open('model.pkl', 'rb') as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "with open('scaler.pkl', 'rb') as file:\n",
    "    scaler = pickle.load(file)\n",
    "\n",
    "\n",
    "st.title(\"Predicción de Aceptación de Producto Bancario\")\n",
    "\n",
    "\n",
    "st.write(\"\"\"\n",
    "Ingrese los datos del cliente para predecir si aceptará o no el producto.\n",
    "\"\"\")\n",
    "\n",
    "\n",
    "age = st.number_input(\"Edad del cliente:\", min_value=18, max_value=100, step=1)\n",
    "balance = st.number_input(\"Balance promedio en la cuenta bancaria:\")\n",
    "housing = st.selectbox(\"¿Tiene hipoteca?\", options=['Sí', 'No'])\n",
    "loan = st.selectbox(\"¿Tiene préstamo personal?\", options=['Sí', 'No'])\n",
    "poutcome = st.selectbox(\"Resultado de la campaña previa:\", options=['Éxito', 'Fracaso', 'Desconocido'])\n",
    "\n",
    "\n",
    "\n",
    "housing_map = {'No': 0, 'Sí': 1}\n",
    "loan_map = {'No': 0, 'Sí': 1}\n",
    "poutcome_map = {'Desconocido': 0, 'Fracaso': 1, 'Éxito': 2}\n",
    "\n",
    "\n",
    "input_data = np.array([[age, balance, housing_map[housing], loan_map[loan], poutcome_map[poutcome]]])\n",
    "input_data_scaled = scaler.transform(input_data)\n",
    "\n",
    "\n",
    "if st.button(\"Predecir\"):\n",
    "    prediction = model.predict(input_data_scaled)\n",
    "    resultado = \"Aceptará el producto\" if prediction[0] == 1 else \"No aceptará el producto\"\n",
    "    st.subheader(f\"Resultado de la predicción: {resultado}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "80660bbb-e94b-44a5-8312-f8bfe85b3670",
   "metadata": {},
   "outputs": [],
   "source": []
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
