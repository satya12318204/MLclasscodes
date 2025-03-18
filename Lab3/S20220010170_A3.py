import pandas as pd
import numpy as np

data = pd.read_csv('Lab_A3_Admission_Predict_Ver1.1.csv')

data = data.drop(columns=['Serial No.'])

features = data.drop(columns=['Chance of Admit'])
target = data['Chance of Admit']

features = np.c_[np.ones(features.shape[0]), features]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

X_train_array = np.array(X_train)
y_train_array = np.array(y_train)
X_test_array = np.array(X_test)
y_test_array = np.array(y_test)

class CustomRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X_transpose = X.T
        self.coefficients = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        return X @ self.coefficients

model = CustomRegression()
model.fit(X_train_array, y_train_array)

train_predictions = model.predict(X_train_array)
test_predictions = model.predict(X_test_array)

def compute_mse(true_values, predicted_values):
    return np.mean((true_values - predicted_values) ** 2)

def compute_r2(true_values, predicted_values):
    total_variance = np.sum((true_values - np.mean(true_values)) ** 2)
    residual_variance = np.sum((true_values - predicted_values) ** 2)
    return 1 - (residual_variance / total_variance)

training_mse = compute_mse(y_train_array, train_predictions)
testing_mse = compute_mse(y_test_array, test_predictions)
training_r2 = compute_r2(y_train_array, train_predictions)
testing_r2 = compute_r2(y_test_array, test_predictions)

print(f"Training Mean Squared Error: {training_mse:.4f}")
print(f"Testing Mean Squared Error: {testing_mse:.4f}")
print(f"Training R² Score: {training_r2:.4f}")
print(f"Testing R² Score: {testing_r2:.4f}")

candidate_data = np.array([1, 320, 110, 3, 4.5, 4.0, 9.0, 1])  
predicted_admission_chance = model.predict(candidate_data.reshape(1, -1))
print(f"Predicted Admission Chance for New Candidate: {predicted_admission_chance[0]:.4f}")
