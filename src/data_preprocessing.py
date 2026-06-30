import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load data
df = pd.read_csv('data/heart.csv')

# Features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Categorical columns
cat_cols = ['Sex', 'ChestPainType', 'RestingECG',
            'ExerciseAngina', 'ST_Slope']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X[cat_cols],
    y,
    test_size=0.2,
    random_state=42
)

# One Hot Encoding
encoder = OneHotEncoder(drop='first', sparse_output=False)

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Save encoder
pickle.dump(encoder, open('models/encoder.pkl', 'wb'))