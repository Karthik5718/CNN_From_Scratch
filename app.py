import streamlit as st
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict
from PIL import Image
import cv2
st.title("Digit Classifier (0 vs 1)")
@st.cache_resource
def build_and_train():
    def preprocess_data(x, y, limit):
        zero_index = np.where(y == 0)[0][:limit]
        one_index = np.where(y == 1)[0][:limit]
        all_indices = np.hstack((zero_index, one_index))
        all_indices = np.random.permutation(all_indices)
        x, y = x[all_indices], y[all_indices]
        x = x.reshape(len(x), 1, 28, 28)
        x = x.astype("float32") / 255
        y = to_categorical(y)
        y = y.reshape(len(y), 2, 1)
        return x, y
    (x_train, y_train), _ = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 200)
    network = [
        Convolutional((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]
    train(
        network,
        binary_cross_entropy,
        binary_cross_entropy_prime,
        x_train,
        y_train,
        epochs=10,
        learning_rate=0.1
    )
    return network
network = build_and_train()
uploaded_file = st.file_uploader("Upload Digit Image (0 or 1)", type=["png", "jpg", "jpeg"])
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
