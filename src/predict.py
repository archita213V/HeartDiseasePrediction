import pickle
import pandas as pd

# Load model and encoder
model = pickle.load(open('models/heart_disease_model.pkl', 'rb'))
encoder = pickle.load(open('models/encoder.pkl', 'rb'))

# Input values
sample = pd.DataFrame({
    'Sex': ['M'],
    'ChestPainType': ['ATA'],
    'RestingECG': ['Normal'],
    'ExerciseAngina': ['N'],
    'ST_Slope': ['Up']
})

# Encode input
sample_encoded = encoder.transform(sample)

# Predict
prediction = model.predict(sample_encoded)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")
    