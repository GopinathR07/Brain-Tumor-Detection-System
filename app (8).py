
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI scan to detect tumor using Deep Learning")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction Function
# -------------------------------
def predict(image):
    IMG_SIZE = 224

    # Resize image
    img = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    return prediction

# -------------------------------
# Main Execution
# -------------------------------
if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    st.write("🔍 Analyzing image...")

    # Get prediction
    result = predict(image)

    # -------------------------------
    # Display Result
    # -------------------------------
    if result > 0.5:
        st.error("🚨 Tumor Detected")
        st.metric("Confidence", f"{result*100:.2f}%")
        st.progress(int(result * 100))
    else:
        st.success("✅ No Tumor Detected")
        st.metric("Confidence", f"{(1-result)*100:.2f}%")
        st.progress(int((1-result) * 100))
