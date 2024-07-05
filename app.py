# CODE IN APP.PY

import streamlit as st
import joblib
import numpy as np

# Load the model, scaler, and PCA
model = joblib.load('fire_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

st.title("Forest Fire Prediction App")

# Input fields for the indices
DSR = st.number_input('DSR (0-64)', min_value=0.00, max_value=64.00, value=0.00)
FWI = st.number_input('FWI (0-64)', min_value=0.00, max_value=64.00, value=0.00)
ISI = st.number_input('ISI (0-30)', min_value=0.00, max_value=30.00, value=0.00)
DC = st.number_input('DC (0-1043)', min_value=0.00, max_value=1043.00, value=0.00)
DMC = st.number_input('DMC (0-467)', min_value=0.00, max_value=467.00, value=0.00)
FFMC = st.number_input('FFMC (0-99)', min_value=0.00, max_value=99.00, value=0.00)
BUI = st.number_input('BUI (0-325)', min_value=0.00, max_value=457.00, value=0.00)

# Prediction
if st.button('Predict'):
    features = np.array([[DSR, FWI, ISI, DC, DMC, FFMC, BUI]])
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    prediction = model.predict(features_pca)
    if prediction == 1:
        st.write("Prediction: Fire")
    else:
        st.write("Prediction: No Fire")