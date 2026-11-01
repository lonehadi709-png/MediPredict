
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    data = data.drop_duplicates()
    data = data.fillna(data.mean(numeric_only=True))

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    return X, y


def train_model(X, y):
    # Split data BEFORE scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Fit scaler only on training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    joblib.dump(model, "diabetes_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # Evaluate model
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Training Completed!")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler


def predict_diabetes(input_data):
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Preserve feature names
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    input_df = pd.DataFrame(input_data, columns=feature_names)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        return "The person is likely to have Diabetes."
    else:
        return "The person is not likely to have Diabetes."


if __name__ == "__main__":
    X, y = load_and_preprocess_data("diabetes.csv")

    model, scaler = train_model(X, y)

    test_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]

    result = predict_diabetes(test_data)

    print("\nTest Prediction:")
    print(result)
