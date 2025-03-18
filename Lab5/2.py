import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('Lab_A3_Admission_Predict_Ver1.1.csv')

# Drop the 'Serial No.' column as it's not a feature
data = data.drop(columns=['Serial No.'])

# Features and target
X = data.drop(columns=['Chance of Admit'])
y = data['Chance of Admit'].values

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=None, alpha=0.1):
        """
        Initializes Linear Regression model with gradient descent, supporting L1 or L2 regularization.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.m = None
        self.c = None

    def _cost_function(self, X, y, m, c):
        """
        Computes the cost function (MSE) with optional L1 or L2 regularization.
        """
        n = len(y)
        y_pred = np.dot(X, m) + c
        error = y_pred - y
        cost = (1/(2*n)) * np.sum(error**2)

        if self.regularization == 'L2':  # Ridge regularization
            cost += (self.alpha / (2*n)) * np.sum(m**2)
        elif self.regularization == 'L1':  # Lasso regularization
            cost += (self.alpha / (2*n)) * np.sum(np.abs(m))

        return cost

    def fit(self, X, y):
        """
        Fit the model to training data using gradient descent and regularization.
        """
        n, features = X.shape
        self.m = np.zeros(features)
        self.c = 0

        self.cost_history = []

        for i in range(self.iterations):
            y_pred = np.dot(X, self.m) + self.c
            error = y_pred - y

            # Gradients
            dm = (1/n) * np.dot(X.T, error)
            dc = (1/n) * np.sum(error)

            # Apply regularization
            if self.regularization == 'L2':
                dm += (self.alpha / n) * self.m  # L2 regularization
            elif self.regularization == 'L1':
                dm += (self.alpha / n) * np.sign(self.m)  # L1 regularization

            # Update parameters
            self.m -= self.learning_rate * dm
            self.c -= self.learning_rate * dc

            # Record cost
            cost = self._cost_function(X, y, self.m, self.c)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Predict outcomes for new data.
        """
        return np.dot(X, self.m) + self.c

    def get_params(self):
        """
        Get the model parameters (weights and bias).
        """
        return self.m, self.c


def run_gradient_descent(X_train_scaled, X_test_scaled, y_train, y_test, regularization_type, iterations_list):
    """
    Runs gradient descent for given iterations and regularization type, returning train and test errors.
    """
    train_errors = []
    test_errors = []

    print(f"\nRunning Gradient Descent with {regularization_type} regularization")
    
    for iterations in iterations_list:
        model = LinearRegressionGD(learning_rate=0.01, iterations=iterations, regularization=regularization_type, alpha=0.1)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Calculate train and test error (MSE)
        train_mse = np.mean((y_train_pred - y_train)**2)
        test_mse = np.mean((y_test_pred - y_test)**2)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

        # Get slope (m) and bias (c)
        m, c = model.get_params()
        print(f"Iterations: {iterations}, Slope (m): {m}, Bias (c): {c}")
    
    return train_errors, test_errors


# Iteration counts for gradient descent
iterations_list = [50, 100, 150, 200]

# Run for L2 regularization (Ridge)
train_errors_l2, test_errors_l2 = run_gradient_descent(X_train_scaled, X_test_scaled, y_train, y_test, regularization_type='L2', iterations_list=iterations_list)

# Run for L1 regularization (Lasso)
train_errors_l1, test_errors_l1 = run_gradient_descent(X_train_scaled, X_test_scaled, y_train, y_test, regularization_type='L1', iterations_list=iterations_list)

# Plotting train and test errors for L2 (Ridge) and L1 (Lasso) regularization
plt.plot(iterations_list, train_errors_l2, label='Train Error (L2)', marker='o', color='blue')
plt.plot(iterations_list, test_errors_l2, label='Test Error (L2)', marker='o', color='green')
plt.plot(iterations_list, train_errors_l1, label='Train Error (L1)', marker='x', color='red')
plt.plot(iterations_list, test_errors_l1, label='Test Error (L1)', marker='x', color='orange')
plt.title('Train and Test Error vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
