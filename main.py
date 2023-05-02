import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from neuralnetwork import NeuralNetwork

        
best_dropout = 0.8
base_model = NeuralNetwork(best_dropout).to('cpu')
# Reading the file and storing it in dataframe.
filep = pd.read_csv('dataset.csv', header=0)
df = pd.DataFrame(filep)

update_f1 = df['f1'] == 'c'
df.loc[update_f1, 'f1'] = 0
update_f2 = df['f2'] == 'f'
df.loc[update_f2, 'f2'] = 0
update_f4 = df['f4'] == 'a'
df.loc[update_f4, 'f4'] = 0
update_f5 = df['f5'] == 'b'
df.loc[update_f5, 'f5'] = 0
update_f6 = df['f6'] == 'd'
df.loc[update_f6, 'f6'] = 0
update_f7 = df['f7'] == 'e'
df.loc[update_f7, 'f7'] = 0

#Dropping all the NaN values
df = df.dropna()

# Creating a Y_target and X_data
x_data = df.drop(['target'], axis=1)
y_target = df['target']

# Scaling numerical variables to have zero mean and unit variance.
scaler = StandardScaler()
scaled = scaler.fit_transform(x_data)
x_target_scaled_df = pd.DataFrame(scaled, columns=x_data.columns)
x_target_scaled_df
# Splitting the Datset into train test
x_train, x_test, y_train, y_test = train_test_split(x_target_scaled_df, y_target, train_size=0.8, test_size=0.2)

# Converting the Training Data to Numpy Array
x_train = x_train.to_numpy()
y_train = y_train.to_numpy().reshape(-1,1)

# Converting the Testing Data to Numpy Array
x_test = x_test.to_numpy()
y_test = y_test.to_numpy().reshape(-1,1)

# Converting the Data to tensors.
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

# Define the hyperparameters: Number of epoch, learning Rate, Batch Size
batch_size = 64
learning_rate = 0.01

#Loss Function: We have opted for Binary Cross Entropy Loss
loss_function = nn.BCELoss()
optimizer = torch.optim.Adagrad(base_model.parameters(), lr=learning_rate)
base_model.train(optimizer, 50, x_train, y_train, batch_size, loss_function)
# Saving model to disk
pickle.dump(base_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
int_features = [6,148,72,35,0,33.6,0.627]
final_features = np.array(int_features)
final_feature_tensors = torch.from_numpy(final_features).float()
y_pred = model.test(final_feature_tensors)
print(y_pred.item())