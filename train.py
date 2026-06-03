
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model A: Classical ML
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)

# -------------------------------
# Model B: Deep Learning
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dl_model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

dl_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

dl_model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=16,
    verbose=0
)

loss, dl_accuracy = dl_model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("Deep Learning Accuracy:", dl_accuracy)

# -------------------------------
# Compare Models
# -------------------------------
if rf_accuracy >= dl_accuracy:
    joblib.dump(rf_model, "best_model.pkl")
    best_model = "Random Forest"
    best_accuracy = rf_accuracy
else:
    dl_model.save("best_model.h5")
    best_model = "Deep Learning"
    best_accuracy = dl_accuracy

print(f"Best Model: {best_model}")
print(f"Best Accuracy: {best_accuracy}")

# -------------------------------
# Generate README
# -------------------------------
with open("README.md", "w") as f:
    f.write("# MLOps Model Comparison\n\n")
    f.write(f"Random Forest Accuracy: {rf_accuracy:.4f}\n\n")
    f.write(f"Deep Learning Accuracy: {dl_accuracy:.4f}\n\n")
    f.write(f"Best Model: {best_model}\n")
