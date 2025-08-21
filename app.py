import streamlit as st
import pandas as pd
import joblib

# Load models and scaler
knn = joblib.load("knn_model.pkl")
nb = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load training column names
with open("X_columns.txt", "r") as f:
    training_columns = f.read().splitlines()

# Preprocessing
def preprocess_input(input_data):
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_data[col] = input_data[col].apply(lambda x: 1 if str(x).lower() in ['yes', 'male'] else 0)

    multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaymentMethod']
    input_data = pd.get_dummies(input_data, columns=multi_class_cols, drop_first=True)

    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[training_columns]
    return scaler.transform(input_data)

# Prediction
def predict(model_choice, input_dict):
    input_df = pd.DataFrame([input_dict])
    try:
        processed_input = preprocess_input(input_df)
        model = knn if model_choice == "KNN" else nb
        pred = model.predict(processed_input)[0]
        return "Churn" if pred == 1 else "No Churn"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# App UI
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn using KNN or Naive Bayes.")

# Sidebar for model choice
model_choice = st.sidebar.selectbox("Choose Model", ["KNN", "Naive Bayes"])

# Input form
with st.form("prediction_form"):
    gender = st.radio("Gender", ["Male", "Female"])
    senior = st.radio("Senior Citizen", [0, 1])
    partner = st.radio("Partner", ["Yes", "No"])
    dependents = st.radio("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.radio("Phone Service", ["Yes", "No"])
    paperless = st.radio("Paperless Billing", ["Yes", "No"])
    monthly = st.number_input("Monthly Charges", 0.0)
    total = st.number_input("Total Charges", 0.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_bak = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_supp = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    multiline = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    submit = st.form_submit_button("Predict")

    if submit:
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "PaperlessBilling": paperless,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "PaymentMethod": payment,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak,
            "DeviceProtection": device_protect,
            "TechSupport": tech_supp,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "MultipleLines": multiline
        }

        result = predict(model_choice, input_dict)
        st.success(f"Prediction: {result}")

        # Store history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({**input_dict, "Prediction": result})

# History
st.markdown("## 🕘 Prediction History")
if "history" in st.session_state:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("No predictions made yet.")
