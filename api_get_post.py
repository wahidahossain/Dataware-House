# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:50:26 2020

@author: Nafi-Wafi
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)

@app.route("/predict", methods=['GET','POST']) 
#use decorator pattern for the route
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to dress model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('C:/Users/Nafi-Wafi/Downloads/model_lr2.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('C:/Users/Nafi-Wafi/Downloads/model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)
