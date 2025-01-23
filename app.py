{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4c746786-77d5-4bb0-87ae-d1d7f535245e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-01-23 20:54:51.365 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\aprats\\AppData\\Roaming\\Python\\Python312\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-01-23 20:54:51.368 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
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
    "if age is None or balance is None:\n",
    "    st.error(\"Por favor, ingrese valores válidos para la edad y el balance.\")\n",
    "else:\n",
    "    try:\n",
    "        input_data = np.array([[age, balance, education_map[education], housing_map[housing], loan_map[loan]]])\n",
    "        input_data_df = pd.DataFrame(input_data, columns=['age', 'balance', 'education', 'housing', 'loan'])\n",
    "\n",
    "    \n",
    "        input_data_scaled = scaler.transform(input_data_df)\n",
    "\n",
    "        if st.button(\"Predecir\"):\n",
    "            \n",
    "            prediction = model.predict(input_data_scaled)\n",
    "            resultado = \"Aceptará el producto\" if prediction[0] == 1 else \"No aceptará el producto\"\n",
    "            st.subheader(f\"Resultado de la predicción: {resultado}\")\n",
    "\n",
    "    except KeyError as e:\n",
    "        st.error(f\"Error en el mapeo de valores categóricos: {e}\")\n",
    "    except Exception as e:\n",
    "        st.error(f\"Error al preparar los datos: {e}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ee39124-42dd-47e5-a175-bae369e58815",
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
