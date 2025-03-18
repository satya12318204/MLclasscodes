import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mean = 2
std_dev = 0.5
num_test_samples = 100
n_values = range(100, 1001, 100)
num_datasets = 10

def generate_data(n, mean, std_dev, num_test_samples):
    X_train = np.random.normal(mean, std_dev, n).reshape(-1, 1)
    Y_train = 2 * X_train
    X_test = np.random.normal(mean, std_dev, num_test_samples).reshape(-1, 1)
    Y_test = 2 * X_test
    return X_train, Y_train, X_test, Y_test

def calculate_bias_variance(X_train, Y_train, X_test, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_test_pred = model.predict(X_test)
    
    Y_test_true = 2 * X_test
    bias_squared = np.mean((np.mean(Y_test_pred) - Y_test_true.flatten()) ** 2)
    variance = np.mean(np.var(Y_test_pred, axis=0))
    
    return bias_squared, variance

def perform_analysis(n_values, num_datasets, mean, std_dev, num_test_samples):
    bias_list = []
    variance_list = []

    for n in n_values:
        bias_squared_list = []
        variance_list_temp = []
        
        for _ in range(num_datasets):
            X_train, Y_train, X_test, Y_test = generate_data(n, mean, std_dev, num_test_samples)
            bias_squared, variance = calculate_bias_variance(X_train, Y_train, X_test, Y_test)
            bias_squared_list.append(bias_squared)
            variance_list_temp.append(variance)
        
        bias_list.append(np.mean(bias_squared_list))
        variance_list.append(np.mean(variance_list_temp))
    
    return bias_list, variance_list

bias_list, variance_list = perform_analysis(n_values, num_datasets, mean, std_dev, num_test_samples)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(n_values, bias_list, marker='o', linestyle='-', color='blue')
plt.xlabel('Training Set Size (n)')
plt.ylabel('Bias Squared')
plt.title('n vs Bias Squared')

plt.subplot(1, 2, 2)
plt.plot(n_values, variance_list, marker='o', linestyle='-', color='red')
plt.xlabel('Training Set Size (n)')
plt.ylabel('Variance')
plt.title('n vs Variance')

plt.tight_layout()
plt.show()
