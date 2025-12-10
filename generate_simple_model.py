import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

print("Loading and preparing data...")

# Load the dataset
file_path = 'engine_data 2.csv'
df = pd.read_csv(file_path)

# Print data info
print("\nDataset Info:")
print(df.info())
print("\nSample data:")
print(df.head())

# Define features and labels (using exact column names from CSV)
feature_columns = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[feature_columns].values
y = df['Engine Condition'].values

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
    X_scaled, y, test_size=0.2, random_state=42
)

# Build a simple sequential model
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining model...")
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Test prediction on sample data
print("\nTesting prediction on sample data...")
sample_data = np.array([[3.5, 7.0, 2.5, 77.0, 80.0]], dtype=np.float32)
print("Sample input:", sample_data)
sample_scaled = scaler.transform(sample_data)
print("Scaled input:", sample_scaled)
prediction = model.predict(sample_scaled, verbose=0)
print("Sample prediction:", prediction[0][0])

# Save the model and scaler
print("\nSaving model and scaler...")
os.makedirs('models', exist_ok=True)
model.save('models/simple_engine_model.keras')
joblib.dump(scaler, 'models/scaler.pkl')

print("\nModel and scaler saved successfully")

# Save the value ranges for reference
value_ranges = {
    'feature_names': feature_columns,
    'min_values': X.min(axis=0).tolist(),
    'max_values': X.max(axis=0).tolist(),
    'mean_values': X.mean(axis=0).tolist(),
    'std_values': X.std(axis=0).tolist()
}

with open('models/value_ranges.json', 'w') as f:
    json.dump(value_ranges, f, indent=2) 