import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('Lab_A3_Admission_Predict_Ver1.1.csv')

data = data.drop(columns=['Serial No.'])

X = data.drop(columns=['Chance of Admit'])
y = data['Chance of Admit'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def cost_function(X, y, theta, reg_type=None, alpha=0.1):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1/(2*m)) * np.sum(error**2)

    if reg_type == 'L2':  
        cost += (alpha / (2*m)) * np.sum(theta[1:]**2)  
    elif reg_type == 'L1': 
        cost += (alpha / (2*m)) * np.sum(np.abs(theta[1:]))

    return cost


def gradient_descent(X, y, theta, learning_rate, iterations, reg_type=None, alpha=0.1):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        error = predictions - y

        if reg_type == 'L2':
            gradient = (1/m) * X.T.dot(error) + (alpha / m) * np.concatenate(([0], theta[1:]))
        elif reg_type == 'L1':
            gradient = (1/m) * X.T.dot(error) + (alpha / m) * np.concatenate(([0], np.sign(theta[1:])))
        else:
            gradient = (1/m) * X.T.dot(error)

        theta -= learning_rate * gradient
        cost_history[i] = cost_function(X, y, theta, reg_type, alpha)

    return theta, cost_history


def run_experiment(X_train, X_test, y_train, y_test, reg_type, iterations_list):
    train_errors = []
    test_errors = []

    print(f"\nRunning Gradient Descent with {reg_type} regularization")
    
    X_train_with_intercept = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_with_intercept = np.c_[np.ones(X_test.shape[0]), X_test]

    for iterations in iterations_list:
        theta = np.zeros(X_train_with_intercept.shape[1])
        theta, cost_history = gradient_descent(
            X_train_with_intercept, y_train, theta, learning_rate=0.01, iterations=iterations, reg_type=reg_type, alpha=0.1)

        y_train_pred = X_train_with_intercept.dot(theta)
        y_test_pred = X_test_with_intercept.dot(theta)

        train_mse = np.mean((y_train_pred - y_train)**2)
        test_mse = np.mean((y_test_pred - y_test)**2)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

        print(f"For Iteration {iterations}:")
        print(f"Train Error: {train_mse:.12f}\t Test Error: {test_mse:.12f}")
        print(f"Slope: {', '.join(f'{val:.8f}' for val in theta[1:])}\t Bias: {theta[0]:.8f}\n")

    return train_errors, test_errors


iterations_list = [50, 100, 150, 200]

train_errors_l2, test_errors_l2 = run_experiment(X_train_scaled, X_test_scaled, y_train, y_test, reg_type='L2', iterations_list=iterations_list)

train_errors_l1, test_errors_l1 = run_experiment(X_train_scaled, X_test_scaled, y_train, y_test, reg_type='L1', iterations_list=iterations_list)

plt.plot(iterations_list, train_errors_l2, label='Train Error (L2)', marker='o', color='blue')
plt.plot(iterations_list, test_errors_l2, label='Test Error (L2)', marker='o', color='green')
plt.plot(iterations_list, train_errors_l1, label='Train Error (L1)', marker='x', color='red')
plt.plot(iterations_list, test_errors_l1, label='Test Error (L1)', marker='x', color='orange')
plt.title('Train and Test Error vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')    
plt.legend()
plt.show()
