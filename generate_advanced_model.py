import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objects as go

from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os
import json

print("Loading and preparing data...")

# Load the dataset
file_path = 'engine_data 2.csv'
df = pd.read_csv(file_path)

# Column-wise Analysis with Different Charts
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def plot_column_analysis(df):
    for column in df.columns:
        plt.figure()
        if column == "Engine rpm":
            sns.lineplot(data=df[column], color='purple')
            plt.title(f'Line Plot of {column}')
        elif column == "Lub oil pressure":
            sns.boxplot(x=df[column], color='orange')
            plt.title(f'Box Plot of {column}')
        elif column == "Fuel pressure":
            sns.histplot(df[column], kde=True, color='skyblue', bins=30)
            plt.title(f'Histogram of {column}')
        elif column == "Coolant pressure":
            sns.violinplot(x=df[column], color='lightgreen')
            plt.title(f'Violin Plot of {column}')
        elif column == "lub oil temp":
            sns.stripplot(x=df[column], color='brown')
            plt.title(f'Strip Plot of {column}')
        elif column == "Coolant temp":
            sns.scatterplot(x=range(len(df[column])), y=df[column], color='red')
            plt.title(f'Scatter Plot of {column}')
        elif column == "Engine Condition":
            df[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            plt.ylabel('')
            plt.title(f'Pie Chart of {column}')
        plt.tight_layout()
        plt.savefig(f'assets/plots/{column.lower().replace(" ", "_")}.png')
        plt.close()

# Create plots directory if it doesn't exist
os.makedirs('assets/plots', exist_ok=True)
print("\nGenerating visualizations...")
plot_column_analysis(df)
print("Visualizations saved in assets/plots/")

# Define features and labels
feature_columns = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[feature_columns].values
y = df["Engine Condition"].values

# Print initial class distribution
print("\nInitial class distribution:")
print(Counter(y))

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
print("\nApplying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("After SMOTE:", Counter(y_resampled))

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

print("\nBuilding ensemble model...")
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

print("\nTraining model...")
# Train the model
history = model.fit(
    [X_train_reshaped, X_train_reshaped, X_train_reshaped], y_train, epochs=30, batch_size=32,
    validation_data=([X_test_reshaped, X_test_reshaped, X_test_reshaped], y_test),
    callbacks=[early_stopping], verbose=1
)

# Make predictions
print("\nEvaluating model...")
y_pred_prob = model.predict([X_test_reshaped, X_test_reshaped, X_test_reshaped])
best_threshold = 0.5
y_pred = (y_pred_prob > best_threshold).astype(int)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model artifacts
print("\nSaving model artifacts...")
os.makedirs('models', exist_ok=True)

# Save the ensemble model
model.save('models/advanced_engine_model.keras')
joblib.dump(scaler, 'models/advanced_scaler.pkl')

# Save the value ranges for reference
value_ranges = {
    'feature_names': feature_columns,
    'min_values': X.min(axis=0).tolist(),
    'max_values': X.max(axis=0).tolist(),
    'mean_values': X.mean(axis=0).tolist(),
    'std_values': X.std(axis=0).tolist()
}

with open('models/advanced_value_ranges.json', 'w') as f:
    json.dump(value_ranges, f, indent=2)

print("\nModel and artifacts saved successfully")

# Save training history
history_dict = {
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy']
}

with open('models/training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)

print("\nTraining history saved successfully") 