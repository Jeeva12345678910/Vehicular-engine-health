import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib
import os


file_path = "engine_data 2.csv"
df = pd.read_csv(file_path)


feature_columns = [col for col in df.columns if col not in ["Engine rpm", "Engine Condition"]]
X = df[feature_columns].values
y = df["Engine Condition"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


os.makedirs('models', exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved to models/scaler.pkl")


model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), 
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.2, 
                    callbacks=[early_stopping], 
                    verbose=1)


test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")


model.save("models/engine_health_model.keras")
print("Model saved to models/engine_health_model.keras")