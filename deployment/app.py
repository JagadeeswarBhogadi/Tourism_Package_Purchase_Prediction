import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Jagadesswar/tourism_prediction_model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Product  Purchase Prediction")

st.write("""
This application predicts the likelihood of a tourism package acceptance based on previous history,
Please enter the details of data data below to get a prediction.
""")

st.subheader("Enter customer details:")

# Input widgets
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Email", "Phone", "Social Media"])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
occupation = st.selectbox("Occupation", ["Student", "Employed", "Self-employed", "Retired"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
no_of_person_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=1)
no_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
product_pitched = st.selectbox("Product Pitched", ["Travel Package", "Hotel Booking", "Flight Booking"])
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
number_of_trips = st.number_input("Number of Trips", min_value=0, max_value=10, value=1)
passport = st.selectbox("Has Passport", ["Yes", "No"])
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10, value=5)
own_car = st.selectbox("Own Car", ["Yes", "No"])
no_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Manager", "Developer", "Sales", "HR", "Other"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000)


# Create DataFrame
user_input = pd.DataFrame({
    'Age': [age],
    'TypeofContact': [typeofcontact],
    'CityTier': [citytier],
    'DurationOfPitch': [durationofpitch],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [no_of_person_visiting],
    'NumberOfFollowups': [no_of_followups],
    'ProductPitched': [product_pitched],
    'PreferredPropertyStar': [preferred_property_star],
    'MaritalStatus': [marital_status],
    'NumberOfTrips': [number_of_trips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [no_of_children_visiting],
    'Designation': [designation],
    'MonthlyIncome': [monthly_income]
})

# Prediction button
if st.button("Predict"):
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.success("🎉 The customer is likely to purchase the product!")
    else:
        st.warning("❌ The customer is unlikely to purchase the product.")

