import torch
import pandas as pd
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.ip_layer1 = nn.Linear(7, 128)
        self.activation_layer1 = nn.ELU() 
        self.hidden_layer2 = nn.Linear(128, 128)
        self.activation_layer2 = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer3 = nn.Linear(128, 128)
        self.activation_layer3 = nn.ELU()
        self.op_layer4 = nn.Linear(128, 1)
        self.activation_layer4 = nn.Sigmoid()
    
    # Feed Forward Function.
    def forward(self, x):
        x = self.ip_layer1(x)
        x = self.activation_layer1(x)
        x = self.hidden_layer2(x)
        x = self.activation_layer2(x)
        x = self.dropout(x)
        x = self.hidden_layer3(x)
        x = self.activation_layer3(x)
        x = self.op_layer4(x)
        x = self.activation_layer4(x)
        return x
    
    def train(self,optimizer, num_epochs,x_train,y_train,batch_size,loss_function):
        for epoch in range(num_epochs):
            # Training the Neural Network
            for i in range(0, x_train.shape[0], batch_size):
                # Forward pass
                y_hat = self.forward(x_train[i:i+batch_size])
                trainloss = loss_function(y_hat, y_train[i:i+batch_size])

                # Backward and optimize
                optimizer.zero_grad()
                trainloss.backward()
                optimizer.step()
        
        
    def test(self,x_test):
         # Testing the Neural Network
        with torch.no_grad():
            y_pred = self.forward(x_test)
            
        return y_pred