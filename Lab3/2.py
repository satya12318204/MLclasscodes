import numpy as np
import pandas as pd

# Function to normalize features
def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Function to split data into train and test sets (70% train, 30% test)
def train_test_split(X, y, test_size=0.3):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Function to perform Gradient Descent
def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(iterations):
        gradient = (1/m) * np.dot(X.T, np.dot(X, theta) - y)
        theta -= learning_rate * gradient
    return theta

# Function to predict the target variable
def predict(X, theta):
    return np.dot(X, theta)

# Function to calculate Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to calculate R2 Score
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Main function
def main():
    # Load dataset using pandas
    data = pd.read_csv("Lab_A3_Admission_Predict_Ver1.1.csv")
    
    # Prepare features and target
    X = data.iloc[:, 1:-1].values  # All columns except the first and last one
    y = data.iloc[:, -1].values   # Last column

    # Normalize features
    X = normalize_features(X)

    # Add intercept term
    X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones to X

    # Train-test split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Set hyperparameters
    learning_rate = 0.01
    iterations = 100

    # Train the model using Gradient Descent
    theta = gradient_descent(X_train, y_train, learning_rate, iterations)

    # Make predictions
    y_train_pred = predict(X_train, theta)
    y_test_pred = predict(X_test, theta)

    # Calculate and print errors
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    print(f"Train Error: {train_error}")
    print(f"Test Error: {test_error}")

    # Calculate and print MSE and R2 score
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train R2 Score: {train_r2}")
    print(f"Test R2 Score: {test_r2}")

if __name__ == "__main__":
    main()
