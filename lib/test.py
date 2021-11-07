#!/usr/bin/env python
# coding: utf-8


from torchvision import datasets, transforms
import numpy as np
import qiskit
from quantum_c2c.quantum_c2c import quantum_c2c,AutoEncoder,Hybrid
import os

# training data of autoencoder (pre-training)
# DataLoader
X_train_autoencoder = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train_autoencoder.targets == 0)[0], 
                np.where(X_train_autoencoder.targets == 1)[0])

X_train_autoencoder.data = X_train_autoencoder.data[idx]
X_train_autoencoder.targets = X_train_autoencoder.targets[idx]


# training data of main training
# Concentrating on the first 100 samples
n_samples = 100

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]



# test data of main training

n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]


os.mkdir('model')
trained_encoder_model,y_predicted=quantum_c2c(X_train_autoencoder,X_train,X_test,AutoEncoder(input_shape=X_train.data.shape,encoded_len=16),Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2,encoded_len=16),'model',epochs=1)
print("Successful!")