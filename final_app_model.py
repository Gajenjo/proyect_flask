from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True
name = __name__

# Enruta la landing page (endpoint /)
@app.route("/", methods=['GET'])
def hello():
    return f"Bienvenido a la API de {name}"

# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods=['GET'])
def predict():

    model = pickle.load(open('ad_model.pkl','rb'))
    # ad_model.pkl
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if (tv is None) | (radio is None) | (newspaper is None):
        return "Not enought data to predict"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])
    
    return jsonify({'prediction': prediction[0]})


# Enruta la funcion al endpoint /api/v1/retrain
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():

    data = pd.read_csv('data/Advertising.csv', index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                    data['sales'],
                                                    test_size = 0.20,
                                                    random_state=42)

    model = Lasso(alpha=6000)
    model.fit(X_train, y_train)

    model.fit(data.drop(columns=['sales']), data['sales'])  
    pickle.dump(model, open('ad_model.pkl', 'wb'))

    return "Model retrained. New evaluation metric RMSE: " + str(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

app.run()