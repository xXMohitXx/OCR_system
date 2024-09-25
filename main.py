# main.py

import streamlit as st
import numpy as np
from PIL import Image
from ocr_network import OCRNeuralNetwork
from utils import load_mnist_data, preprocess_image

st.title("Optimized OCR System with Feedforward ANN")

# Load MNIST data
x_train, y_train, x_test, y_test = load_mnist_data()

# Define optimized neural network structure
layer_dims = [784, 128, 64, 32, 10]

# Initialize and train the model
nn = OCRNeuralNetwork(layer_dims)
st.write("Training the model...")
nn.train(x_train, y_train, learning_rate=0.1, num_iterations=1000, print_cost=True)
st.write("Model training completed!")

# User uploads image
uploaded_file = st.file_uploader("Upload an image (28x28 pixels)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Predict the digit
    AL, _ = nn.forward_propagation(processed_image)
    predicted_digit = np.argmax(AL, axis=0)[0]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted digit: {predicted_digit}")
