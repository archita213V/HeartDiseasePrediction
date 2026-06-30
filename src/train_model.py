import pickle
from sklearn.ensemble import RandomForestClassifier

# Load processed data
from data_preprocessing import X_train, y_train

# Create model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
rf.fit(X_train, y_train)

# Save model
pickle.dump(rf, open('models/heart_disease_model.pkl', 'wb'))

print("Model saved successfully")