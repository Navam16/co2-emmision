
import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('model_compressed.pkl')
le_state = joblib.load('le_state.pkl')
le_industry = joblib.load('le_industry.pkl')
le_unit = joblib.load('le_unit.pkl')

# Streamlit UI
st.title("CO2 Emission Prediction")

# User input for prediction
state_input = st.text_input("State (e.g., CA):")
industry_input = st.text_input("Industry (e.g., 44):")
heat_input = st.number_input("Heat Input (mmBTU/hr):", min_value=0.0)
unit_type_input = st.text_input("Unit Type (e.g., 30):")
ch4_input = st.number_input("Methane (CH4):", min_value=0.0)
n2o_input = st.number_input("N2O:", min_value=0.0)

# Prediction logic
if st.button("Predict CO2 Emission"):
    try:
        # Encode the inputs
        state_encoded = le_state.transform([state_input])[0]
        industry_encoded = le_industry.transform([industry_input])[0]
        unit_type_encoded = le_unit.transform([unit_type_input])[0]

        # Prepare the input for prediction
        input_features = np.array([[state_encoded, industry_encoded, heat_input, unit_type_encoded, ch4_input, n2o_input]])
        
        # Predict CO2 emission
        predicted_co2 = model.predict(input_features)
        
        # Display the result
        st.write(f"🌿 Predicted CO2 Emission: {predicted_co2[0]:.2f} (in units)")

    except ValueError as e:
        st.error(f"⚠️ Error: {str(e)}")
        st.write("Please check your inputs or try using values present in your training dataset.")
