import numpy as np
import torch
from flask import Flask, request, render_template
import pickle
from neuralnetwork import NeuralNetwork

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = np.array(float_features)
    final_feature_tensors = torch.from_numpy(final_features).float()
    y_pred = model.test(final_feature_tensors)
    output = int(y_pred.item())

    return render_template('index.html', prediction_text='The Predicted Value is {}'.format(y_pred.item()))


if __name__ == "__main__":
    app.run(debug=True)