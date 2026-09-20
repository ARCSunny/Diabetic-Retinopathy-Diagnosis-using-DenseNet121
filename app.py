import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input


MODEL_PATH = "diabetic_retinopathy_model.keras"
IMG_SIZE = 224
CLASS_NAMES = ["DR", "No_DR"] 

st.set_page_config(page_title="Diabetic Retinopathy Diagnosis", page_icon="🩺", layout="centered")



@st.cache_resource
def load_model():
    # The model's preprocessing Lambda layer references DenseNet121's
    # preprocess_input function by its original name, so it must be
    # supplied again when loading.
    custom_objects = {"preprocess_input": preprocess_input}
    return tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)


def predict(image: Image.Image, model):
    resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return CLASS_NAMES[predicted_index], float(probabilities[predicted_index]), probabilities



st.title("🩺 Diabetic Retinopathy Diagnosis")
st.write(
    "Upload a retina fundus image. The model (DenseNet121 transfer learning) "
    "will predict whether it shows signs of Diabetic Retinopathy (DR)."
)

with st.spinner("Loading model..."):
    model = load_model()

uploaded_file = st.file_uploader("Choose a retina image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        predicted_class, confidence, probabilities = predict(image, model)

    with col2:
        if predicted_class == "DR":
            st.error(f"### Prediction: {predicted_class}")
        else:
            st.success(f"### Prediction: {predicted_class}")

        st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.write("**Class probabilities:**")
        for class_name, prob in zip(CLASS_NAMES, probabilities):
            st.write(f"{class_name}: {prob * 100:.2f}%")
            st.progress(float(prob))

    st.caption(
        "⚠️ This tool is for educational/demonstration purposes only and is "
        "not a substitute for professional medical diagnosis."
    )
else:
    st.info("Upload an image above to get a prediction.")
