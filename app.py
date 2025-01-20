import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

# Link to download the model from Google Drive
url = 'https://drive.google.com/uc?id=1WBrSKHPPxDmiJ1jR1xKcUgIUoA2gxRwk'  # Replace with your model link
output = 'model.keras'

# Download the model using gdown
gdown.download(url, output, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(output)

# Streamlit app interface
st.title("Digit Recognizer")
st.write("Upload an image of a handwritten digit (28x28 pixels) to predict the digit.")

# Upload an image from the user
img = st.file_uploader("Upload your digit image", type=["jpg", "png", "jpeg"])

if img is not None:
    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        # Preprocess the image
        img = Image.open(img).convert("L")  # Convert the image to grayscale
        img = img.resize((28, 28))  # Resize the image to 28x28 pixels
        img_array = np.array(img) / 255.0  # Normalize pixel values to the range [0, 1]
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape the image for the model

        # Make a prediction using the model
        pred = model.predict(img_array)
        digit = np.argmax(pred)  # Get the predicted digit

        # Display the result
        st.write(f"The model predicts this digit is: **{digit}**")
else:
    st.write("Please upload an image of a handwritten digit.")

# Footer with author information
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 14px; color: gray;">
        This page was created by <strong>Mohamed Hosam</strong>
    </div>
    """, unsafe_allow_html=True
)
