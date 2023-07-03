import pickle
from pickle import dump
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Create the file "scaling.pkl" and add the model to the file
model = pickle.dump(model, open("scaling.pkl", "wb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json
    new_data = scalermodel.transform(np.array(data['data']).reshape(1, -1))
    output = model.predict(new_data)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalermodel.transform(np.array(data).reshape(1, -1))
    output = model.predict(final_input)
    return render_template('home.html', prediction_text="THE PREDICTION VALUE IS {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)
