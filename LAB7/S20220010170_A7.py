import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

random_state = random.randint(0, 100)

data = pd.read_csv('Lab_A7_iris.data.csv')
shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

label_encoder = LabelEncoder()
shuffled_data['species'] = label_encoder.fit_transform(shuffled_data['species'])

features = shuffled_data.drop('species', axis=1)
target = shuffled_data['species']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=random_state, stratify=target
)

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def find_neighbors(train_X, train_y, test_point, k):
    distances = []
    for index in range(len(train_X)):
        distance = calculate_distance(test_point, train_X.iloc[index])
        distances.append((train_y.iloc[index], distance))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

def predict_knn(train_X, train_y, test_point, k):
    neighbors = find_neighbors(train_X, train_y, test_point, k)
    return max(set(neighbors), key=neighbors.count)

def perform_k_fold_validation(X, y, k, num_splits=4):
    fold_size = len(X) // num_splits
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    accuracy_list = []
    
    for split in range(num_splits):
        validation_indices = indices[split * fold_size:(split + 1) * fold_size]
        training_indices = np.concatenate((indices[:split * fold_size], indices[(split + 1) * fold_size:]))
        
        X_train_fold, y_train_fold = X.iloc[training_indices], y.iloc[training_indices]
        X_val_fold, y_val_fold = X.iloc[validation_indices], y.iloc[validation_indices]

        correct_count = 0
        for j in range(len(X_val_fold)):
            prediction = predict_knn(X_train_fold, y_train_fold, X_val_fold.iloc[j], k)
            if prediction == y_val_fold.iloc[j]:
                correct_count += 1
        
        accuracy = correct_count / len(X_val_fold)
        accuracy_list.append(accuracy)

    return np.mean(accuracy_list), np.std(accuracy_list)

odd_k_values = [k for k in range(1, 10) if k % 2 != 0]
optimal_k = odd_k_values[0]
max_accuracy = 0
mean_accuracies = []
std_accuracies = []

print("Results from 4-Fold Cross Validation:")
for k in odd_k_values:
    mean_acc, std_acc = perform_k_fold_validation(X_train, y_train, k)
    mean_accuracies.append(mean_acc)
    std_accuracies.append(std_acc)
    
    print(f"k={k}: Mean Accuracy = {mean_acc:.4f}, Std = {std_acc:.4f}")
    
    if mean_acc > max_accuracy:
        max_accuracy = mean_acc
        optimal_k = k
    elif mean_acc == max_accuracy and k < optimal_k:
        optimal_k = k

print(f"\nOptimal k identified: {optimal_k} with Cross-Validation Accuracy = {max_accuracy:.4f}\n")

predictions = [predict_knn(X_train, y_train, point, optimal_k) for point in X_test.values]
test_set_accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {test_set_accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.errorbar(odd_k_values, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=5, color='b', label='Mean Accuracy ± Std Dev')
plt.axvline(x=optimal_k, color='r', linestyle='--', label='Optimal k')
plt.title('KNN Cross-Validation Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(odd_k_values)
plt.grid(True)
plt.legend()
plt.show()
