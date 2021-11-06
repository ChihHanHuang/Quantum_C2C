import numpy as np
import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
import pickle
import os


###########
#QuantumCircuit
###########

class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        
        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])





class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, backend, shots, shift,encoded_len):
        
        
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(encoded_len, backend, shots)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)




###########
#AutoEncoder
###########

class AutoEncoder(nn.Module):
    
    def __init__(self,input_shape,encoded_len):
        super(AutoEncoder, self).__init__()
        input_len=input_shape[-1]*input_shape[-2]
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, encoded_len),
            nn.Tanh(),
            nn.Linear(encoded_len, 1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1, encoded_len),
            nn.Tanh(),
            nn.Linear(encoded_len, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, input_len),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return codes, decoded


def quantum_c2c(X_train_autoencoder,X_train,X_test,autoencoder_model,quantum_curcuit,saving_folder,epochs=10):
    
    ##pre-train autoencoder
    
    # Settings
    
    batch_size = 128
    lr = 0.008

    #get data for pre-training autoencoder

    train_loader = torch.utils.data.DataLoader(X_train_autoencoder, batch_size=batch_size, shuffle=True)
    input_len=X_train.data.shape[-1]*X_train.data.shape[-2]
    
    # Optimizer and loss function
    
    model = autoencoder_model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    
    # pre-train
    
    print('Pre-training autoencoder')
    for epoch in range(epochs):
        for data, labels in train_loader:
            
            inputs = data.view(-1, input_len)
            codes, decoded = model(inputs)
            optimizer.zero_grad()
            loss = loss_function(decoded, inputs)
            loss.backward()
            optimizer.step()

        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())


    pretraining_autoencoder_path=os.path.join(saving_folder, 'pretraining_autoencoder.pth')
    torch.save(model, pretraining_autoencoder_path)

    autoencoder = torch.load(pretraining_autoencoder_path)
    autoencoder.eval()    
    
    
    ##train the main model
    
    #create data loader
    
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)
    
    #create model
    
    class Net_autoencoder(nn.Module):
        def __init__(self):
            super(Net_autoencoder, self).__init__()
            self.encoder = autoencoder.encoder
            self.decoder = autoencoder.decoder

            self.hybrid = quantum_curcuit

        def forward(self, x):
            inputs = x.view(-1, input_len)
            codes = self.encoder(inputs)
            decoded = self.decoder(codes)
            x = self.hybrid(codes)
            return torch.cat((x, 1 - x), -1),decoded    
    
    model = Net_autoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.NLLLoss()
    autoencoder_loss=nn.MSELoss()

    
    print('Training main model')
    model.train()
    for epoch in range(epochs):
        total_loss = []
        loss_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            output,decoded = model(data)
            # Calculating loss
            data_flatten = data.view(-1, 784)
            loss = loss_func(output, target)+autoencoder_loss(decoded,data_flatten)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()

            total_loss.append(loss.item())
        loss_list.append(sum(total_loss)/len(total_loss))
        print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
            100. * (epoch + 1) / epochs, loss_list[-1]))    
    trained_encoder_model_path=os.path.join(saving_folder, 'trained_encoder_model.pth')
    torch.save(model.encoder, trained_encoder_model_path)
    
    
    
    # test model
    
    model.eval()
    y_predicted=[]
    
    with torch.no_grad():

        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            output,decoded = model(data)

            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            y_predicted.append(pred)
            loss = loss_func(output, target)
            total_loss.append(loss.item())

        print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
            sum(total_loss) / len(total_loss),
            correct / len(test_loader) * 100)
            )    
    

    
    
    y_predicted=np.array(y_predicted)

    
    return model.encoder,y_predicted






