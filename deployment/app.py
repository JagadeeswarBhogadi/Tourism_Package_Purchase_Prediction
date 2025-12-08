import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the full pipeline (preprocessing + model)
model_path = hf_hub_download(repo_id="Jagadesswar/tourism_prediction_model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Product Purchase Prediction
st.title("Tourism Product Purchase Prediction")

st.write("""
This application predicts the likelihood of a tourism package acceptance based on previous history.
Please enter the details of the customer below to get a prediction.
""")

st.subheader("Enter customer details:")

# Mapping dictionaries based on Xtrain encoding
typeofcontact_map = {"Self Enquiry": 0, "Company Invited": 1}
occupation_map = {"Salaried": 2, "Free Lancer": 1, "Small Business": 3, "Large Business": 4}
gender_map = {"Male": 1, "Female": 2, "Other": 3}
product_pitched_map = {"Deluxe": 0, "Basic": 1, "Standard": 2, "Super Deluxe": 3, "King": 4}
marital_status_map = {"Single": 0, "Divorced": 1, "Married": 2, "Unmarried": 3}
passport_map = {"Yes": 1, "No": 0}
own_car_map = {"Yes": 1, "No": 0}
designation_map = {"Manager": 1, "Executive": 2, "Senior Manager": 3, "AVP": 4, "VP": 5}

# Input widgets
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
no_of_person_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=1)
no_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
number_of_trips = st.number_input("Number of Trips", min_value=0, max_value=10, value=1)
passport = st.selectbox("Has Passport", ["Yes", "No"])
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10, value=5)
own_car = st.selectbox("Own Car", ["Yes", "No"])
no_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000)

# Convert all inputs to numeric codes for model
user_input = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact_map[typeofcontact],
    "CityTier": citytier,
    "DurationOfPitch": durationofpitch,
    "Occupation": occupation_map[occupation],
    "Gender": gender_map[gender],
    "NumberOfPersonVisiting": no_of_person_visiting,
    "NumberOfFollowups": no_of_followups,
    "ProductPitched": product_pitched_map[product_pitched],
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status_map[marital_status],
    "NumberOfTrips": number_of_trips,
    "Passport": passport_map[passport],
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car_map[own_car],
    "NumberOfChildrenVisiting": no_of_children_visiting,
    "Designation": designation_map[designation],
    "MonthlyIncome": monthly_income
}])

# Prediction button
if st.button("Predict"):
    # Use the pre-loaded and pre-fitted pipeline to predict directly (no need to fit again)
    prediction = model.predict(user_input)[0]  # Directly use the model for prediction
    if prediction == 1:
        st.write("🎯 Final Prediction:", prediction)
        st.success("🎉 The customer is likely to purchase the product!")
    else:
        st.write("🎯 Final Prediction:", prediction)
        st.warning("❌ The customer is unlikely to purchase the product")
