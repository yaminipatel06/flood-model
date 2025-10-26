import streamlit as st
import numpy as np
import joblib
import pickle

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
columns = pickle.load(open('columns.pkl','rb'))

st.title("Flood Prediction App")
st.write("Use sliders to enter environmental parameters:")

input_data = []
for col in columns:
    val = st.slider(f"{col}", 0, 10, 5)  # min=0, max=10, default=5
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    occurrence = model.predict(input_scaled)[0]
    predicted_probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Results")
    st.write("**Flood Occurrence:**", "Flood" if occurrence == 1 else "No Flood")
    st.write("**Predicted Flood Probability:**", f"{predicted_probability:.2f}")
    
from google.colab import files

files.download('model.pkl')
files.download('scaler.pkl')
files.download('columns.pkl')
files.download('app.py')
