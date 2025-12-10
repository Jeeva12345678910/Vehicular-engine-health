import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load the dataset
file_path = 'engine_data 2.csv'
df = pd.read_csv(file_path)

# Define features and labels
feature_columns = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[feature_columns].values
y = df['Engine Condition'].values

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Reshape data for input
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)

# Define Model Architectures
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    return Model(inputs, x)

def build_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=False)(inputs)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    return Model(inputs, x)

def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    return Model(inputs, x)

# Build Model
input_shape = (X.shape[1], 1)
cnn_model = build_cnn(input_shape)
lstm_model = build_lstm(input_shape)
transformer_model = build_transformer(input_shape)

combined = Concatenate()([cnn_model.output, lstm_model.output, transformer_model.output])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[cnn_model.input, lstm_model.input, transformer_model.input], outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(
    [X_train_reshaped, X_train_reshaped, X_train_reshaped], y_train, epochs=30, batch_size=32,
    validation_data=([X_test_reshaped, X_test_reshaped, X_test_reshaped], y_test),
    callbacks=[early_stopping], verbose=1
)

# Save the model and scaler
os.makedirs('models', exist_ok=True)
model.save('models/engine_health_model.keras')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model and scaler saved successfully.")