import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('jamb_exam_results.csv')

# Create a binary target variable: Pass if JAMB_Score >= 200, else Fail
df['Pass'] = df['JAMB_Score'].apply(lambda x: 1 if x >= 200 else 0)

# Select relevant features (ignoring Student_ID and original JAMB_Score for now)
X = df.drop(columns=['JAMB_Score', 'Pass', 'Student_ID'])
y = df['Pass']

# Convert categorical features to dummy variables (one-hot encoding)
X_encoded = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# --- Task 1: Implement Random Forest from Scratch ---

# Function to calculate Gini index (same as decision tree implementation)
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion * proportion
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Function to split dataset based on an attribute and attribute value (same as decision tree)
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Function to choose the best split (same as decision tree but limited to max_features)
def get_best_split(dataset, max_features):
    class_values = list(set(row[-1] for row in dataset))
    features = np.random.choice(range(len(dataset[0])-1), max_features, replace=False)
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Function to create a terminal node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Function to split a node or create a terminal node (same as decision tree)
def split(node, max_depth, min_size, depth, max_features):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left, max_features)
        split(node['left'], max_depth, min_size, depth+1, max_features)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right, max_features)
        split(node['right'], max_depth, min_size, depth+1, max_features)

# Function to build a decision tree (similar to the previous decision tree code but limited features)
def build_tree(train, max_depth, min_size, max_features):
    root = get_best_split(train, max_features)
    split(root, max_depth, min_size, 1, max_features)
    return root

# Function to make predictions (same as decision tree)
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Function to create bootstrap samples
def bootstrap_sample(X_train, y_train):
    n_samples = len(X_train)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X_train[indices], y_train[indices]

# Function to build a random forest
def random_forest(X_train, y_train, max_depth, min_size, max_features, n_trees):
    forest = []
    for _ in range(n_trees):
        X_samp, y_samp = bootstrap_sample(X_train, y_train)
        train_data = np.column_stack((X_samp, y_samp)).tolist()
        tree = build_tree(train_data, max_depth, min_size, max_features)
        forest.append(tree)
    return forest

# Function to make predictions with the forest (majority voting)
def random_forest_predict(forest, row):
    predictions = [predict(tree, row) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

# --- Train the Random Forest ---

n_trees = 10
max_depth = 5
min_size = 10
max_features = int(np.sqrt(X_train.shape[1]))

# Convert training set to NumPy arrays
X_train_np = X_train.values
y_train_np = y_train.values

# Build the forest
forest = random_forest(X_train_np, y_train_np, max_depth, min_size, max_features, n_trees)

# Make predictions on the test set
test_data = np.column_stack((X_test.values, y_test.values)).tolist()
rf_predictions = [random_forest_predict(forest, row) for row in test_data]

# --- Task 3: Evaluate Performance ---
accuracy_rf = accuracy_score(y_test, rf_predictions)
precision_rf = precision_score(y_test, rf_predictions)
recall_rf = recall_score(y_test, rf_predictions)
f1_rf = f1_score(y_test, rf_predictions)
conf_matrix_rf = confusion_matrix(y_test, rf_predictions)

print(f"Random Forest -> Accuracy: {accuracy_rf:.3f}, Precision: {precision_rf:.3f}, Recall: {recall_rf:.3f}, F1-Score: {f1_rf:.3f}")
print("Confusion Matrix:\n", conf_matrix_rf)

# Convert the full training set into a list of lists for single decision tree
train_data = np.column_stack((X_train.values, y_train.values)).tolist()

# Build a single decision tree using the full training set (without bootstrap sampling)
clf_single = build_tree(train_data, max_depth=5, min_size=10, max_features=X_train.shape[1])

# Make predictions with the single decision tree
single_tree_predictions = [predict(clf_single, row) for row in test_data]

# Evaluate the single decision tree
accuracy_single = accuracy_score(y_test, single_tree_predictions)
precision_single = precision_score(y_test, single_tree_predictions)
recall_single = recall_score(y_test, single_tree_predictions)
f1_single = f1_score(y_test, single_tree_predictions)

print(f"Single Decision Tree -> Accuracy: {accuracy_single:.3f}, Precision: {precision_single:.3f}, Recall: {recall_single:.3f}, F1-Score: {f1_single:.3f}")

# --- Task 5: Visualization of Performance ---
n_trees_list = [1, 5, 10, 20, 50]
accuracy_list = []

for n in n_trees_list:
    forest = random_forest(X_train_np, y_train_np, max_depth, min_size, max_features, n)
    rf_predictions = [random_forest_predict(forest, row) for row in test_data]
    accuracy_rf = accuracy_score(y_test, rf_predictions)
    accuracy_list.append(accuracy_rf)

plt.plot(n_trees_list, accuracy_list, marker='o')
plt.title('Random Forest Accuracy vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
