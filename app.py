{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "be170f87-14c8-4417-93e9-254ab878c2c9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-01-22 16:43:48.416 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\aprats\\AppData\\Roaming\\Python\\Python312\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-01-22 16:43:48.420 Session state does not function when running a script without `streamlit run`\n",
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
    "\n",
    "input_data = np.array([[age, balance, education_map[education], housing_map[housing], loan_map[loan]]])\n",
    "input_data_scaled = scaler.transform(input_data)\n",
    "\n",
    "\n",
    "if st.button(\"Predecir\"):\n",
    "    prediction = model.predict(input_data_scaled)\n",
    "    resultado = \"Aceptará el producto\" if prediction[0] == 1 else \"No aceptará el producto\"\n",
    "    st.subheader(f\"Resultado de la predicción: {resultado}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "90cac63a-7142-45c9-b8c0-d43c29abf0ef",
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
