# utils.py

import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten and normalize the data
    x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0

    # One-hot encoding for labels
    y_train_encoded = np.eye(10)[y_train].T
    y_test_encoded = np.eye(10)[y_test].T

    return x_train, y_train_encoded, x_test, y_test_encoded

def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).reshape(784, 1) / 255.0
    return image
