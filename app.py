{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ab07c71e-8730-4bf8-a15e-a46b4af5e9a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "13e7bdb4-08f1-4b3f-9be3-8318771cd340",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-01-22 15:53:40.848 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\aprats\\AppData\\Roaming\\Python\\Python312\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-01-22 15:53:40.852 Session state does not function when running a script without `streamlit run`\n",
      "C:\\Users\\aprats\\AppData\\Local\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "X has 6 features, but StandardScaler is expecting 5 features as input.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 29\u001b[0m\n\u001b[0;32m     26\u001b[0m poutcome_map \u001b[38;5;241m=\u001b[39m {\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mDesconocido\u001b[39m\u001b[38;5;124m'\u001b[39m: \u001b[38;5;241m0\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mFracaso\u001b[39m\u001b[38;5;124m'\u001b[39m: \u001b[38;5;241m1\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mÉxito\u001b[39m\u001b[38;5;124m'\u001b[39m: \u001b[38;5;241m2\u001b[39m}\n\u001b[0;32m     28\u001b[0m input_data \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39marray([[age, balance, education_map[education], housing_map[housing], loan_map[loan], poutcome_map[poutcome]]])\n\u001b[1;32m---> 29\u001b[0m input_data_scaled \u001b[38;5;241m=\u001b[39m \u001b[43mscaler\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtransform\u001b[49m\u001b[43m(\u001b[49m\u001b[43minput_data\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     32\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m st\u001b[38;5;241m.\u001b[39mbutton(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mPredecir\u001b[39m\u001b[38;5;124m\"\u001b[39m):\n\u001b[0;32m     33\u001b[0m     prediction \u001b[38;5;241m=\u001b[39m model\u001b[38;5;241m.\u001b[39mpredict(input_data_scaled)\n",
      "File \u001b[1;32m~\\AppData\\Local\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_set_output.py:313\u001b[0m, in \u001b[0;36m_wrap_method_output.<locals>.wrapped\u001b[1;34m(self, X, *args, **kwargs)\u001b[0m\n\u001b[0;32m    311\u001b[0m \u001b[38;5;129m@wraps\u001b[39m(f)\n\u001b[0;32m    312\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mwrapped\u001b[39m(\u001b[38;5;28mself\u001b[39m, X, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[1;32m--> 313\u001b[0m     data_to_wrap \u001b[38;5;241m=\u001b[39m \u001b[43mf\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    314\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(data_to_wrap, \u001b[38;5;28mtuple\u001b[39m):\n\u001b[0;32m    315\u001b[0m         \u001b[38;5;66;03m# only wrap the first output for cross decomposition\u001b[39;00m\n\u001b[0;32m    316\u001b[0m         return_tuple \u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m    317\u001b[0m             _wrap_data_with_container(method, data_to_wrap[\u001b[38;5;241m0\u001b[39m], X, \u001b[38;5;28mself\u001b[39m),\n\u001b[0;32m    318\u001b[0m             \u001b[38;5;241m*\u001b[39mdata_to_wrap[\u001b[38;5;241m1\u001b[39m:],\n\u001b[0;32m    319\u001b[0m         )\n",
      "File \u001b[1;32m~\\AppData\\Local\\anaconda3\\Lib\\site-packages\\sklearn\\preprocessing\\_data.py:1045\u001b[0m, in \u001b[0;36mStandardScaler.transform\u001b[1;34m(self, X, copy)\u001b[0m\n\u001b[0;32m   1042\u001b[0m check_is_fitted(\u001b[38;5;28mself\u001b[39m)\n\u001b[0;32m   1044\u001b[0m copy \u001b[38;5;241m=\u001b[39m copy \u001b[38;5;28;01mif\u001b[39;00m copy \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcopy\n\u001b[1;32m-> 1045\u001b[0m X \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_validate_data\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   1046\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1047\u001b[0m \u001b[43m    \u001b[49m\u001b[43mreset\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[0;32m   1048\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mcsr\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1049\u001b[0m \u001b[43m    \u001b[49m\u001b[43mcopy\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcopy\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1050\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mFLOAT_DTYPES\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1051\u001b[0m \u001b[43m    \u001b[49m\u001b[43mforce_writeable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[0;32m   1052\u001b[0m \u001b[43m    \u001b[49m\u001b[43mforce_all_finite\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mallow-nan\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1053\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1055\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m sparse\u001b[38;5;241m.\u001b[39missparse(X):\n\u001b[0;32m   1056\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mwith_mean:\n",
      "File \u001b[1;32m~\\AppData\\Local\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:654\u001b[0m, in \u001b[0;36mBaseEstimator._validate_data\u001b[1;34m(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)\u001b[0m\n\u001b[0;32m    651\u001b[0m     out \u001b[38;5;241m=\u001b[39m X, y\n\u001b[0;32m    653\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m no_val_X \u001b[38;5;129;01mand\u001b[39;00m check_params\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mensure_2d\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mTrue\u001b[39;00m):\n\u001b[1;32m--> 654\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_check_n_features\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mreset\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mreset\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    656\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m out\n",
      "File \u001b[1;32m~\\AppData\\Local\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:443\u001b[0m, in \u001b[0;36mBaseEstimator._check_n_features\u001b[1;34m(self, X, reset)\u001b[0m\n\u001b[0;32m    440\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m\n\u001b[0;32m    442\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m n_features \u001b[38;5;241m!=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_features_in_:\n\u001b[1;32m--> 443\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    444\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mX has \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mn_features\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m features, but \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__class__\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    445\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mis expecting \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_features_in_\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m features as input.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    446\u001b[0m     )\n",
      "\u001b[1;31mValueError\u001b[0m: X has 6 features, but StandardScaler is expecting 5 features as input."
     ]
    }
   ],
   "source": [
    "with open('model.pkl', 'rb') as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "with open('scaler.pkl', 'rb') as file:\n",
    "    scaler = pickle.load(file)\n",
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
    "poutcome = st.selectbox(\"Resultado de la campaña previa:\", options=['Éxito', 'Fracaso', 'Desconocido'])\n",
    "\n",
    "\n",
    "education_map = {'Primario': 0, 'Secundario': 1, 'Terciario': 2}\n",
    "housing_map = {'No': 0, 'Sí': 1}\n",
    "loan_map = {'No': 0, 'Sí': 1}\n",
    "poutcome_map = {'Desconocido': 0, 'Fracaso': 1, 'Éxito': 2}\n",
    "\n",
    "input_data = np.array([[age, balance, education_map[education], housing_map[housing], loan_map[loan], poutcome_map[poutcome]]])\n",
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
   "id": "c5c98e4b-7970-4164-9863-67468b73f1db",
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
