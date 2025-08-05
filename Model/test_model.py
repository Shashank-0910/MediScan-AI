
import joblib
import numpy as np

model = joblib.load('parkinson_model.pkl')
scaler = joblib.load('scaler.pkl')

sample_input = np.array([[119.992, 157.302, 74.997, 0.00784, 0.00007,
                          0.00370, 0.00554, 0.01109, 0.04374, 0.426,
                          0.02182, 0.03130, 0.02971, 0.06425, 0.02211,
                          21.033, 0.414783, 0.815285, -4.813031, 0.266482,
                          2.301442, 0.284654]])

scaled_input = scaler.transform(sample_input)

prediction = model.predict(scaled_input)

print(" Parkinson Prediction:", "Positive" if prediction[0] == 1 else "Negative")
