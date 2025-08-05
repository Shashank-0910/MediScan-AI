from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

try:
    parkinson_model = joblib.load('parkinson_model.pkl')
    parkinson_scaler = joblib.load('parkinson_scaler.pkl')
    
    diabetes_model = joblib.load('diabetes_model.pkl')
    diabetes_scaler = joblib.load('diabetes_scaler.pkl')
    
    heart_model = joblib.load('heart_disease_model.pkl')
    heart_scaler = joblib.load('heart_disease_scaler.pkl')

    print(" Models and scalers loaded successfully.")
except Exception as e:
    print(f" Error loading models or scalers: {e}")
    exit()

@app.route('/')
def home():
    return "Welcome to the Disease Prediction API!"

@app.route('/predict-parkinson', methods=['POST'])
def predict_parkinson():
    try:
        data = request.json  
        print(f"Received data: {data}")  

        required_keys = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                         'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
                         'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
                         'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
                         'Spread1', 'Spread2', 'D2', 'PPE']

        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required input data"}), 400

        input_data = np.array([list(data.values())])  
        input_scaled = parkinson_scaler.transform(input_data)  

        prediction = parkinson_model.predict(input_scaled) 
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f" Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/predict-diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json  
        print(f"Received data: {data}") 

        required_keys = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required input data"}), 400

        input_data = np.array([list(data.values())])
        input_scaled = diabetes_scaler.transform(input_data)  

        prediction = diabetes_model.predict(input_scaled)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f" Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/predict-heart', methods=['POST'])
def predict_heart():
    try:
        data = request.json  
        print(f"Received data: {data}") 

        required_keys = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs',
                         'restecg', 'thalach', 'exang', 'oldpeak',
                         'slope', 'ca', 'thal']

        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required input data"}), 400

        input_data = np.array([list(data.values())]) 
        input_scaled = heart_scaler.transform(input_data)  

        prediction = heart_model.predict(input_scaled)  
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
