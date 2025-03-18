import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, precision_recall_curve
from scipy.interpolate import interp1d

# Load dataset
data = pd.read_csv('Lab A6_heart_failure_clinical_records_dataset.csv')

# Selected Features
input_columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'ejection_fraction', 
                 'diabetes', 'high_blood_pressure', 'platelets', 'serum_creatinine', 
                 'serum_sodium', 'sex', 'smoking']
X = data[input_columns].values
y = data['DEATH_EVENT'].values

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization function
def normalize(dataset):
    for col in range(dataset.shape[1]):
        mean = np.mean(dataset[:, col])
        std_dev = np.std(dataset[:, col])
        dataset[:, col] = (dataset[:, col] - mean) / std_dev
    return dataset

X_train = normalize(X_train)
X_test = normalize(X_test)

# F1 Score Calculation
def compute_f1_score(actual, predicted):
    true_positive = np.sum((predicted == 1) & (actual == 1))
    false_positive = np.sum((predicted == 1) & (actual == 0))
    false_negative = np.sum((predicted == 0) & (actual == 1))
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

# Logistic Regression Model
class SimpleLogisticRegression:
    def __init__(self):
        self.params = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def prepare(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]  # Adding intercept

    def train(self, X, y, learning_rate=0.001, epochs=10000):
        X_new = self.prepare(X)
        self.params = np.zeros((X_new.shape[1], 1))
        
        for i in range(epochs):
            z = np.dot(X_new, self.params)
            predictions = self.sigmoid(z)
            gradient = np.dot(X_new.T, (predictions - y.reshape(-1, 1))) / y.size
            self.params -= learning_rate * gradient
    
    def predict(self, X):
        X_new = self.prepare(X)
        return (self.sigmoid(np.dot(X_new, self.params)) > 0.5).astype(int).flatten()

# Naive Bayes Classifier
class GaussianNaiveBayes:
    def __init__(self):
        self.means = None
        self.variances = None
        self.class_priors = None
        self.classes = None

    def train(self, X, y):
        self.classes = np.unique(y)
        self.means = np.zeros((len(self.classes), X.shape[1]))
        self.variances = np.zeros((len(self.classes), X.shape[1]))
        self.class_priors = np.zeros(len(self.classes))

        for idx, class_label in enumerate(self.classes):
            X_class = X[y == class_label]
            self.means[idx, :] = np.mean(X_class, axis=0)
            self.variances[idx, :] = np.var(X_class, axis=0)
            self.class_priors[idx] = X_class.shape[0] / X.shape[0]

    def _calculate_probability(self, idx, x):
        mean = self.means[idx]
        variance = self.variances[idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

    def _predict_single(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.class_priors[idx])
            posterior = np.sum(np.log(self._calculate_probability(idx, x)))
            total_posterior = prior + posterior
            posteriors.append(total_posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

# Train Logistic Regression Model
log_model = SimpleLogisticRegression()
log_model.train(X_train, y_train)

# Logistic Regression Predictions
y_pred_train_logistic = log_model.predict(X_train)
y_pred_test_logistic = log_model.predict(X_test)

# Logistic Regression Performance Metrics
f1_train_logistic = compute_f1_score(y_train, y_pred_train_logistic)
f1_test_logistic = compute_f1_score(y_test, y_pred_test_logistic)
precision_test_logistic = precision_score(y_test, y_pred_test_logistic)
recall_test_logistic = recall_score(y_test, y_pred_test_logistic)

# Print Results
print(f"Logistic Regression F1 Score (Train): {f1_train_logistic}")
print(f"Logistic Regression F1 Score (Test): {f1_test_logistic}")
print(f"Logistic Regression Precision (Test): {precision_test_logistic}")
print(f"Logistic Regression Recall (Test): {recall_test_logistic}")

# Confusion Matrix for Logistic Regression
print("\nConfusion Matrix (Logistic Regression - Train):")
print(confusion_matrix(y_train, y_pred_train_logistic))

print("\nConfusion Matrix (Logistic Regression - Test):")
print(confusion_matrix(y_test, y_pred_test_logistic))

# Train Naive Bayes Model
nb_model = GaussianNaiveBayes()
nb_model.train(X_train, y_train)

# Naive Bayes Predictions and Accuracy
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = np.mean(y_pred_nb == y_test)

# Confusion Matrix for Naive Bayes
print(f"\nNaive Bayes Accuracy (Test): {nb_accuracy}")
print("\nConfusion Matrix (Naive Bayes - Test):")
print(confusion_matrix(y_test, y_pred_nb))

# Plot ROC Curve for Logistic Regression
y_score_logistic = log_model.sigmoid(np.dot(log_model.prepare(X_test), log_model.params)).flatten()
fpr_log, tpr_log, _ = roc_curve(y_test, y_score_logistic)

# Remove duplicate values
fpr_unique_log, unique_idx_log = np.unique(fpr_log, return_index=True)
tpr_unique_log = tpr_log[unique_idx_log]

# Smoothing using interpolation
fpr_smooth_log = np.linspace(0, 1, 500)
tpr_smooth_log = interp1d(fpr_unique_log, tpr_unique_log, kind='cubic')(fpr_smooth_log)

plt.figure(figsize=(8, 6))
plt.plot(fpr_smooth_log, tpr_smooth_log, label='ROC - Logistic Regression', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

# Plot Precision-Recall Curve for Logistic Regression
precision_vals_log, recall_vals_log, _ = precision_recall_curve(y_test, y_score_logistic)

# Remove duplicates
recall_unique_log, unique_recall_idx_log = np.unique(recall_vals_log, return_index=True)
precision_unique_log = precision_vals_log[unique_recall_idx_log]

# Interpolate precision-recall for smoother curve
recall_smooth_log = np.linspace(0, 1, 500)
precision_smooth_log = interp1d(recall_unique_log, precision_unique_log, kind='cubic')(recall_smooth_log)

plt.figure(figsize=(8, 6))
plt.plot(recall_smooth_log, precision_smooth_log, label='Precision-Recall - Logistic Regression', color='green')
plt.title("Precision-Recall Curve - Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()
plt.show()
