import streamlit as st
import pandas as pd
import joblib

#  Load model and tools
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("🚗 Insurance Fraud Detection App")
st.write("Provide the claim details below to predict if it's fraud or not.")


# Define input form
with st.form("fraud_detection_form"):
    col1, col2, col3 = st.columns(3)

    # Collect user inputs in proper formats
    with col1:
        Month = st.selectbox("Month of Accident", options=label_encoders["Month"].classes_)
        DayOfWeek = st.selectbox("Day of Week of Accident", options=label_encoders["DayOfWeek"].classes_)
        AccidentArea = st.selectbox("Accident Area", options=label_encoders["AccidentArea"].classes_)
        DayOfWeekClaimed = st.selectbox("Day of Week Claimed", options=label_encoders["DayOfWeekClaimed"].classes_)
        MonthClaimed = st.selectbox("Month Claimed", options=label_encoders["MonthClaimed"].classes_)

    with col2:
        WeekOfMonth = st.number_input("Week of Month", min_value=1, max_value=5, value=1)
        WeekOfMonthClaimed = st.number_input("Week of Month Claimed", min_value=1, max_value=5, value=1)
        Age = st.number_input("Age of Policyholder", min_value=18, max_value=100, value=30)
        RepNumber = st.number_input("Rep Number", min_value=1, max_value=50, value=1)
        Deductible = st.number_input("Deductible", min_value=0, max_value=1000, value=500)

    with col3:
        PastNumberOfClaims = st.selectbox("Past Number of Claims", options=label_encoders["PastNumberOfClaims"].classes_)
        AgeOfVehicle = st.selectbox("Age of Vehicle", options=label_encoders["AgeOfVehicle"].classes_)
        PoliceReportFiled = st.selectbox("Police Report Filed", options=label_encoders["PoliceReportFiled"].classes_)
        WitnessPresent = st.selectbox("Witness Present", options=label_encoders["WitnessPresent"].classes_)

    # Submit button
    submitted = st.form_submit_button("Predict Fraud")

#  Make prediction
if submitted:
    # Prepare input data
    user_input = {
        "Month": Month,
        "WeekOfMonth": WeekOfMonth,
        "DayOfWeek": DayOfWeek,
        "AccidentArea": AccidentArea,
        "DayOfWeekClaimed": DayOfWeekClaimed,
        "MonthClaimed": MonthClaimed,
        "WeekOfMonthClaimed": WeekOfMonthClaimed,
        "Age": Age,
        "RepNumber": RepNumber,
        "Deductible": Deductible,
        "PastNumberOfClaims": PastNumberOfClaims,
        "AgeOfVehicle": AgeOfVehicle,
        "PoliceReportFiled": PoliceReportFiled,
        "WitnessPresent": WitnessPresent
    }

    # Encode categorical features
    for col in label_encoders:
        if col in user_input:
            encoder = label_encoders[col]
            if user_input[col] not in encoder.classes_:
                st.error(f"Invalid input for {col}. Please choose from {list(encoder.classes_)}")
                st.stop()
            user_input[col] = encoder.transform([user_input[col]])[0]

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Scale numeric features
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ This claim is predicted as: **FRAUD**")
    else:
        st.success(f"✅ This claim is predicted as: **NOT FRAUD**")

    # Show probability
    st.write(f"Prediction Confidence: Fraud: {prediction_proba[1]:.2f} | Not Fraud: {prediction_proba[0]:.2f}")

    # Optional: Show input summary
    with st.expander("See input details"):
        st.json(user_input)

# streamlit run app.py
#python -m streamlit run app.py
