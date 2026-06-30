from data_preprocessing import X_test, y_test
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Load model
model = pickle.load(open('models/heart_disease_model.pkl', 'rb'))

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Report
print(classification_report(y_test, y_pred))