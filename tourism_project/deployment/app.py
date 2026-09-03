import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('tourism_project/deployment/model.joblib')

st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="centered")
st.title('Predict Customer Purchase for Wellness Tourism Package')

st.markdown("--- Say Hello to the Future of Travel Sales! ---")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Input fields for customer details, matching the X_train columns
with st.form("prediction_form"):
    st.header("Customer Profile")
    age = st.slider('Age', min_value=18, max_value=80, value=35)
    typeofcontact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
    citytier = st.selectbox('City Tier', [1, 2, 3])
    durationofpitch = st.slider('Duration of Pitch (minutes)', min_value=1, max_value=60, value=10)
    occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Housewife', 'Other', 'Free Lancer'])
    gender = st.selectbox('Gender', ['Male', 'Female', 'Fe Male'])
    numberofpersonvisiting = st.slider('Number of Persons Visiting', min_value=1, max_value=6, value=2)
    numberoffollowups = st.slider('Number of Follow-ups', min_value=0, max_value=6, value=2)
    productpitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    preferredpropertystar = st.selectbox('Preferred Property Star Rating', [3.0, 4.0, 5.0])
    maritalstatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
    numberoftrips = st.slider('Number of Trips Annually', min_value=1, max_value=20, value=2)
    passport = st.radio('Has Passport?', ['Yes', 'No'])
    owncar = st.radio('Owns a Car?', ['Yes', 'No'])
    numberofchildrenvisiting = st.slider('Number of Children Visiting', min_value=0, max_value=4, value=0)
    designation = st.selectbox('Designation', ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
    monthlyincome = st.number_input('Monthly Income (USD)', min_value=5000.0, max_value=100000.0, value=25000.0, step=100.0)
    pitchsatisfactionscore = st.slider('Pitch Satisfaction Score (1-5)', min_value=1, max_value=5, value=3)

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Map 'Yes'/'No' to 1/0 for binary features
        passport_val = 1 if passport == 'Yes' else 0
        owncar_val = 1 if owncar == 'Yes' else 0

        # Create a DataFrame from the inputs
        input_data = pd.DataFrame([[age,
                                      typeofcontact,
                                      citytier,
                                      durationofpitch,
                                      occupation,
                                      gender,
                                      numberofpersonvisiting,
                                      numberoffollowups,
                                      productpitched,
                                      preferredpropertystar,
                                      maritalstatus,
                                      numberoftrips,
                                      passport_val,
                                      owncar_val,
                                      numberofchildrenvisiting,
                                      designation,
                                      monthlyincome,
                                      pitchsatisfactionscore]],
                                    columns=['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
                                             'Occupation', 'Gender', 'NumberOfPersonVisiting',
                                             'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
                                             'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar',
                                             'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
                                             'PitchSatisfactionScore'])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        st.subheader("Prediction Results")
        if prediction == 1:
            st.success("This customer is LIKELY to purchase the Wellness Tourism Package!")
            st.balloons()
        else:
            st.info("This customer is UNLIKELY to purchase the Wellness Tourism Package.")

        st.write(f"Probability of Purchase: {prediction_proba[1]:.2f}")
        st.write(f"Probability of Not Purchasing: {prediction_proba[0]:.2f}")
