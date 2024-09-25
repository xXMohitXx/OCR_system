# ocr_network.py

import numpy as np

class OCRNeuralNetwork:
    def __init__(self, layer_dims):
        np.random.seed(1)
        self.parameters = self.initialize_parameters(layer_dims)
        self.layer_dims = layer_dims

    def initialize_parameters(self, layer_dims):
        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        return parameters

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return np.where(Z <= 0, 0, 1)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def forward_propagation(self, X):
        cache = {'A0': X}
        L = len(self.parameters) // 2
        for l in range(1, L):
            Z = np.dot(self.parameters['W' + str(l)], cache['A' + str(l-1)]) + self.parameters['b' + str(l)]
            A = self.relu(Z)
            cache['Z' + str(l)] = Z
            cache['A' + str(l)] = A
        ZL = np.dot(self.parameters['W' + str(L)], cache['A' + str(L-1)]) + self.parameters['b' + str(L)]
        AL = self.softmax(ZL)
        cache['Z' + str(L)] = ZL
        cache['A' + str(L)] = AL
        return AL, cache

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        return np.squeeze(cost)

    def backward_propagation(self, Y, cache):
        grads = {}
        L = len(self.parameters) // 2
        m = Y.shape[1]

        dZL = cache['A' + str(L)] - Y
        grads['dW' + str(L)] = np.dot(dZL, cache['A' + str(L-1)].T) / m
        grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m

        for l in reversed(range(1, L)):
            dA = np.dot(self.parameters['W' + str(l + 1)].T, dZL)
            dZ = dA * self.relu_derivative(cache['Z' + str(l)])
            grads['dW' + str(l)] = np.dot(dZ, cache['A' + str(l-1)].T) / m
            grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
            dZL = dZ
        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    def train(self, X, Y, learning_rate=0.01, num_iterations=1000, batch_size=64, print_cost=False):
        m = X.shape[1]
        for i in range(num_iterations):
            shuffled_indices = np.random.permutation(m)
            X_shuffled = X[:, shuffled_indices]
            Y_shuffled = Y[:, shuffled_indices]
            for batch_start in range(0, m, batch_size):
                batch_end = batch_start + batch_size
                X_batch = X_shuffled[:, batch_start:batch_end]
                Y_batch = Y_shuffled[:, batch_start:batch_end]

                AL, cache = self.forward_propagation(X_batch)
                cost = self.compute_cost(AL, Y_batch)
                grads = self.backward_propagation(Y_batch, cache)
                self.update_parameters(grads, learning_rate)

            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
