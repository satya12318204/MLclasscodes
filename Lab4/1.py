import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define parameters
train_sizes = range(100, 1100, 100)  # Training set sizes: 100, 200, ..., 1000
test_size = 100  # Fixed test set size
actual_slope = 2  # True relationship: Y = 2X
num_trials = 100  # Number of trials to estimate bias and variance

# Create a test dataset independent of the training sets
np.random.seed(42)
X_test = np.random.normal(loc=2, scale=0.5, size=test_size).reshape(-1, 1)
Y_test = actual_slope * X_test

# Initialize lists to store bias^2 and variance for each training set size
bias_list = []
variance_list = []

# Iterate over each training set size
for n in train_sizes:
    all_predictions = []

    # Generate multiple training sets and fit models
    for _ in range(num_trials):
        X_train = np.random.normal(loc=2, scale=0.5, size=n).reshape(-1, 1)
        Y_train = actual_slope * X_train + np.random.normal(loc=0, scale=0.5, size=n).reshape(-1, 1)
        
        # Create and fit the linear model
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, Y_train)
        
        # Predict using the model on the test data
        predictions = lin_reg.predict(X_test)
        all_predictions.append(predictions)
    
    # Convert predictions to a numpy array
    all_predictions = np.array(all_predictions)
    
    # Compute bias^2 as the squared difference between average prediction and actual test data
    mean_prediction = np.mean(all_predictions, axis=0)
    bias_sq = np.mean((mean_prediction - Y_test) ** 2)
    bias_list.append(bias_sq)
    
    # Compute variance as the mean of variances of the predictions
    pred_variance = np.mean(np.var(all_predictions, axis=0))
    variance_list.append(pred_variance)

# Plotting Bias^2 and Variance against training set size
plt.figure(figsize=(12, 6))

# Plot for Bias^2
plt.subplot(1, 2, 1)
plt.plot(train_sizes, bias_list, marker='o', color='blue', label='Bias^2')
plt.title('Training Set Size vs Bias^2')
plt.xlabel('Training Set Size (n)')
plt.ylabel('Bias^2')
plt.grid(True)
plt.legend()

# Plot for Variance
plt.subplot(1, 2, 2)
plt.plot(train_sizes, variance_list, marker='o', color='red', label='Variance')
plt.title('Training Set Size vs Variance')
plt.xlabel('Training Set Size (n)')
plt.ylabel('Variance')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
