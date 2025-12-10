import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
import os
import json

print("Loading and preparing data...")

# Load the dataset
file_path = 'engine_data 2.csv'
df = pd.read_csv(file_path)

# Print initial class distribution
print("\nInitial class distribution:")
print(df['Engine Condition'].value_counts())
print("\nClass distribution percentages:")
print(df['Engine Condition'].value_counts(normalize=True) * 100)

# Separate majority and minority classes
df_majority = df[df['Engine Condition'] == 1]  # Class 1 is majority
df_minority = df[df['Engine Condition'] == 0]  # Class 0 is minority

print(f"\nMajority class (1) samples: {len(df_majority)}")
print(f"Minority class (0) samples: {len(df_minority)}")

# Downsample majority class to match minority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,
                                 n_samples=len(df_minority),
                                 random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([df_majority_downsampled, df_minority])

print("\nBalanced class distribution:")
print(df_balanced['Engine Condition'].value_counts())
print("\nBalanced class distribution percentages:")
print(df_balanced['Engine Condition'].value_counts(normalize=True) * 100)

# Define features and labels
feature_columns = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df_balanced[feature_columns].values
y = df_balanced['Engine Condition'].values

print("\nFeature statistics:")
print(pd.DataFrame(X, columns=feature_columns).describe())

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled feature ranges:")
print("Min:", X_scaled.min(axis=0))
print("Max:", X_scaled.max(axis=0))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Build a model with balanced class weights
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("\nTraining model...")
# Train the model
history = model.fit(X_train, y_train, 
                   epochs=50, 
                   batch_size=32, 
                   validation_split=0.2,
                   verbose=1)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Test prediction on sample data
print("\nTesting predictions on different scenarios...")

test_scenarios = [
    {
        "name": "Normal Operating Conditions",
        "data": [3.3, 6.65, 2.33, 77.64, 78.43]  # Mean values
    },
    {
        "name": "High Pressure Scenario",
        "data": [4.5, 8.0, 3.0, 77.0, 80.0]
    },
    {
        "name": "Low Pressure Scenario",
        "data": [2.0, 5.0, 1.5, 77.0, 80.0]
    }
]

for scenario in test_scenarios:
    sample_data = np.array([scenario["data"]], dtype=np.float32)
    sample_scaled = scaler.transform(sample_data)
    prediction = model.predict(sample_scaled, verbose=0)
    print(f"\n{scenario['name']}:")
    print(f"Input: {scenario['data']}")
    print(f"Prediction: {prediction[0][0]:.4f}")
    print(f"Status: {'Unhealthy' if prediction[0][0] > 0.5 else 'Healthy'}")

# Calculate and print confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype(int)
from sklearn.metrics import confusion_matrix, classification_report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
print("\nSaving model and scaler...")
os.makedirs('models', exist_ok=True)
model.save('models/balanced_engine_model.keras')
joblib.dump(scaler, 'models/balanced_scaler.pkl')

# Save the value ranges for reference
value_ranges = {
    'feature_names': feature_columns,
    'min_values': X.min(axis=0).tolist(),
    'max_values': X.max(axis=0).tolist(),
    'mean_values': X.mean(axis=0).tolist(),
    'std_values': X.std(axis=0).tolist()
}

with open('models/balanced_value_ranges.json', 'w') as f:
    json.dump(value_ranges, f, indent=2)

print("\nModel and scaler saved successfully") 