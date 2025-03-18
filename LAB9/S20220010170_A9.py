import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv('jamb_exam_results.csv')
df['Pass'] = df['JAMB_Score'].apply(lambda x: 1 if x >= 200 else 0)
X = df.drop(columns=['JAMB_Score', 'Pass', 'Student_ID'])
y = df['Pass']
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

def gini(groups, classes):
    n_instances = sum([len(group) for group in groups])
    gini_score = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion ** 2
        gini_score += (1 - score) * (size / n_instances)
    return gini_score

def split_dataset(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def find_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = float('inf'), float('inf'), float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_dataset(index, row[index], dataset)
            gini_score = gini(groups, class_values)
            if gini_score < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini_score, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def terminal_node(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def build_tree(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = terminal_node(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = terminal_node(left), terminal_node(right)
        return
    if len(left) <= min_size:
        node['left'] = terminal_node(left)
    else:
        node['left'] = find_best_split(left)
        build_tree(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = terminal_node(right)
    else:
        node['right'] = find_best_split(right)
        build_tree(node['right'], max_depth, min_size, depth+1)

def predict_instance(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_instance(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_instance(node['right'], row)
        else:
            return node['right']
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print(f"{'|  ' * depth}Split: [X{node['index']} < {node['value']}]")
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print(f"{'|  ' * depth}Predict: {node}")




train_data = np.column_stack((X_train.values, y_train.values)).tolist()
decision_tree = find_best_split(train_data)
build_tree(decision_tree, max_depth=5, min_size=10, depth=1)

test_data = np.column_stack((X_test.values, y_test.values)).tolist()
predictions = [predict_instance(decision_tree, row) for row in test_data]

print("Custom Decision Tree Structure:")
print_tree(decision_tree)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Custom Decision Tree -> Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")


clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred_sklearn = clf.predict(X_test)

accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
precision_sklearn = precision_score(y_test, y_pred_sklearn)
recall_sklearn = recall_score(y_test, y_pred_sklearn)
f1_sklearn = f1_score(y_test, y_pred_sklearn)

print(f"Scikit-learn Decision Tree -> Accuracy: {accuracy_sklearn:.3f}, Precision: {precision_sklearn:.3f}, Recall: {recall_sklearn:.3f}, F1-Score: {f1_sklearn:.3f}")

plt.figure(figsize=(60,20))
tree.plot_tree(clf, feature_names=X_train.columns, class_names=['Fail', 'Pass'], filled=True)
plt.show()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('jamb_exam_results.csv')
df['Pass'] = df['JAMB_Score'].apply(lambda x: 1 if x >= 200 else 0)
X = df.drop(columns=['JAMB_Score', 'Pass', 'Student_ID'])
y = df['Pass']
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

def gini(groups, classes):
    n_instances = sum([len(group) for group in groups])
    gini_score = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion ** 2
        gini_score += (1 - score) * (size / n_instances)
    return gini_score

def split_dataset(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def find_best_split(dataset, max_features):
    class_values = list(set(row[-1] for row in dataset))
    features = np.random.choice(range(len(dataset[0])-1), max_features, replace=False)
    best_index, best_value, best_score, best_groups = float('inf'), float('inf'), float('inf'), None
    for index in features:
        for row in dataset:
            groups = split_dataset(index, row[index], dataset)
            gini_score = gini(groups, class_values)
            if gini_score < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini_score, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def terminal_node(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def build_tree(node, max_depth, min_size, depth, max_features):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = terminal_node(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = terminal_node(left), terminal_node(right)
        return
    if len(left) <= min_size:
        node['left'] = terminal_node(left)
    else:
        node['left'] = find_best_split(left, max_features)
        build_tree(node['left'], max_depth, min_size, depth+1, max_features)
    if len(right) <= min_size:
        node['right'] = terminal_node(right)
    else:
        node['right'] = find_best_split(right, max_features)
        build_tree(node['right'], max_depth, min_size, depth+1, max_features)

def predict_instance(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_instance(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_instance(node['right'], row)
        else:
            return node['right']

def bootstrap_sample(X_train, y_train):
    n_samples = len(X_train)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X_train[indices], y_train[indices]

def random_forest(X_train, y_train, max_depth, min_size, max_features, n_trees):
    forest = []
    for _ in range(n_trees):
        X_samp, y_samp = bootstrap_sample(X_train, y_train)
        train_data = np.column_stack((X_samp, y_samp)).tolist()
        tree = find_best_split(train_data, max_features)
        build_tree(tree, max_depth, min_size, depth=1, max_features=max_features)
        forest.append(tree)
    return forest

def random_forest_predict(forest, row):
    predictions = [predict_instance(tree, row) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

n_trees = 10
max_depth = 5
min_size = 10
max_features = int(np.sqrt(X_train.shape[1]))

X_train_np = X_train.values
y_train_np = y_train.values
forest = random_forest(X_train_np, y_train_np, max_depth, min_size, max_features, n_trees)

test_data = np.column_stack((X_test.values, y_test.values)).tolist()
rf_predictions = [random_forest_predict(forest, row) for row in test_data]

accuracy_rf = accuracy_score(y_test, rf_predictions)
precision_rf = precision_score(y_test, rf_predictions)
recall_rf = recall_score(y_test, rf_predictions)
f1_rf = f1_score(y_test, rf_predictions)
conf_matrix_rf = confusion_matrix(y_test, rf_predictions)

print(f"Random Forest -> Accuracy: {accuracy_rf:.3f}, Precision: {precision_rf:.3f}, Recall: {recall_rf:.3f}, F1-Score: {f1_rf:.3f}")
print("Confusion Matrix:\n", conf_matrix_rf)

train_data = np.column_stack((X_train.values, y_train.values)).tolist()
clf_single = find_best_split(train_data, max_features=X_train.shape[1])
build_tree(clf_single, max_depth=5, min_size=10, depth=1, max_features=X_train.shape[1])

single_tree_predictions = [predict_instance(clf_single, row) for row in test_data]
accuracy_single = accuracy_score(y_test, single_tree_predictions)
precision_single = precision_score(y_test, single_tree_predictions)
recall_single = recall_score(y_test, single_tree_predictions)
f1_single = f1_score(y_test, single_tree_predictions)

print(f"Single Decision Tree -> Accuracy: {accuracy_single:.3f}, Precision: {precision_single:.3f}, Recall: {recall_single:.3f}, F1-Score: {f1_single:.3f}")

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
