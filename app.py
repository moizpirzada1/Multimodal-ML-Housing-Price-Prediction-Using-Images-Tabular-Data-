import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------------
# 1. Load Model & Scaler
# -------------------------------
st.title("🏡 Multimodal Housing Price Prediction")

# Load model from Hugging Face (compile=False to avoid deserialization errors)
model_path = hf_hub_download(
    repo_id="sunnypirzada/multimodal-housing-price",
    filename="multimodal_house_price.h5"
)
model = tf.keras.models.load_model(model_path, compile=False)

# Load scaler locally from repo
try:
    scaler = joblib.load("scaler.pkl")  # scaler.pkl must be in the same folder as this file
    st.success("✅ Scaler loaded successfully")
except Exception as e:
    scaler = None
    st.warning(f"⚠️ Scaler not loaded: {e}")

# -------------------------------
# 2. Initialize History
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# 3. User Inputs
# -------------------------------
st.header("Enter House Features")

bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=2000)
lot_size = st.number_input("Lot Size", min_value=500, max_value=20000, value=5000)
age = st.number_input("House Age (years)", min_value=0, max_value=100, value=10)

tab_data = np.array([[bedrooms, bathrooms, sqft, lot_size, age]])
if scaler is not None:
    tab_data = scaler.transform(tab_data)

st.subheader("Upload House Image")
uploaded_file = st.file_uploader("Choose a house image", type=["jpg", "png", "jpeg"])

# -------------------------------
# 4. Prediction
# -------------------------------
if st.button("Predict Price"):
    if uploaded_file is not None:
        # Process image
        img = load_img(uploaded_file, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict([img_array, tab_data])
        price = float(pred[0][0])

        # Save to history
        st.session_state.history.append({
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "SqFt": sqft,
            "Lot Size": lot_size,
            "Age": age,
            "Predicted Price": price
        })

        st.success(f"🏠 Estimated Price: ${price:,.2f}")
    else:
        st.warning("⚠️ Please upload a house image first.")

# -------------------------------
# 5. Show History
# -------------------------------
if st.session_state.history:
    st.header("📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)

    # Download option
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
    )
