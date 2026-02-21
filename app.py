import streamlit as st
import numpy as np
import pickle
from PIL import Image
import cv2
from network import predict
st.title("Digit Classifier (0 vs 1)")
with open("cnn_model.pkl", "rb") as f:
    network = pickle.load(f)
uploaded_file = st.file_uploader("Upload a digit image (0 or 1)", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)
    image = cv2.resize(image, (28, 28))
    image = 255 - image
    image = image.astype("float32") / 255
    image = image.reshape(1, 28, 28)
    output = predict(network, image)
    prediction = np.argmax(output)
    st.image(image, width=150, clamp=True)
    st.success(f"Predicted Digit: {prediction}")
