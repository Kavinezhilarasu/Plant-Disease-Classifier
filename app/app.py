import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# Set up the page config to look modern
st.set_page_config(
    page_title="Plant Disease Classifier",
    layout="centered"
)

# Set up paths relative to current execution context
# Assuming the app is run from the root of the project with `streamlit run app/app.py`
MODEL_PATH = 'models/plant_disease_model.keras'
CLASS_NAMES_PATH = 'models/class_names.json'

@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return None, None
        
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
        
    return model, class_names

def preprocess_image(image, target_size=(224, 224)):
    # Convert image to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize
    image = image.resize(target_size)
    # Convert to array
    img_array = np.array(image)
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("🌿 Plant Disease Classifier")
    st.write("Upload an image of a plant leaf and let the deep learning model identify the disease!")
    
    model, class_names = load_model_and_classes()
    
    if model is None or class_names is None:
        st.error("Model or class names not found. Please run the `train.py` script first to train the model and generate these files.")
        st.info("You can train a quick version by checking inside `train.py` that `USE_SUBSET` is set to `True`.")
        return
        
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Predicting...")
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)[0]
        
        # Get the top prediction
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = h_confidence = float(predictions[predicted_idx])
        
        # Format the output beautifully
        st.markdown(f"### Prediction: **{predicted_class.replace('_', ' ').title()}**")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
        
        # Show top 3 predictions as a progress bar for context
        st.write("---")
        st.write("Top 3 Predictions Details:")
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        
        for idx in top_3_indices:
            label = class_names[idx].replace('_', ' ').title()
            conf = float(predictions[idx])
            st.write(f"{label}: {conf*100:.1f}%")
            st.progress(conf)

if __name__ == '__main__':
    main()
