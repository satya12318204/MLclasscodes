import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print('Executing Part 1')

def create_separable_data(num_samples=20, separation_dist=2.0, seed=0):
    np.random.seed(seed)
    
    negative_mean = np.array([-separation_dist, -separation_dist])
    covariance_matrix = np.array([[1.0, 0.5], [0.5, 1.0]]) 
    X_neg = np.random.multivariate_normal(negative_mean, covariance_matrix, num_samples)
    y_neg = -np.ones(num_samples)
    
    positive_mean = np.array([separation_dist, separation_dist])
    X_pos = np.random.multivariate_normal(positive_mean, covariance_matrix, num_samples)
    y_pos = np.ones(num_samples)
    
    features = np.vstack((X_neg, X_pos))
    labels = np.hstack((y_neg, y_pos))
    
    return features, labels

features, labels = create_separable_data(num_samples=20, separation_dist=3.0, seed=42)

def svm_using_quadratic_programming(features, labels):
    num_samples, num_features = features.shape
    
    kernel_matrix = np.dot(features, features.T)
    
    P = matrix(np.outer(labels, labels) * kernel_matrix)
    q = matrix(-np.ones(num_samples))
    G = matrix(-np.eye(num_samples))
    h = matrix(np.zeros(num_samples))
    A = matrix(labels.astype(float), (1, num_samples))
    b = matrix(0.0)
    
    solvers.options['show_progress'] = False
    
    solution = solvers.qp(P, q, G, h, A, b)
    
    alphas = np.ravel(solution['x'])
    
    support_vectors = alphas > 1e-5
    indices = np.arange(len(alphas))[support_vectors]
    alphas_sv = alphas[support_vectors]
    sv_features = features[support_vectors]
    sv_labels = labels[support_vectors]
    
    weights = np.sum(alphas_sv[:, None] * sv_labels[:, None] * sv_features, axis=0)
    
    bias = np.mean(sv_labels - np.dot(sv_features, weights))
    
    return weights, bias, sv_features, alphas_sv

weights, bias, sv_features, alphas_sv = svm_using_quadratic_programming(features, labels)

def visualize_decision_boundary(features, labels, weights, bias, sv_features, classifier=None, axis=None, title=""):
    if axis is None:
        axis = plt.gca()
    
    axis.scatter(features[labels == -1, 0], features[labels == -1, 1], color='red', label='Class -1')
    axis.scatter(features[labels == 1, 0], features[labels == 1, 1], color='blue', label='Class +1')
    
    if sv_features is not None:
        axis.scatter(sv_features[:, 0], sv_features[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(x_limits[0]-1, x_limits[1]+1, 500),
                         np.linspace(y_limits[0]-1, y_limits[1]+1, 500))
    
    if classifier is None:  
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias
    else:  
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    
    axis.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    axis.set_title(title)
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.grid(True)

def compare_svm_with_sklearn(features, labels, weights_custom, bias_custom):
    classifier = svm.SVC(kernel='linear', C=1e10)  
    classifier.fit(features, labels)
    
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(16, 6))

    visualize_decision_boundary(features, labels, weights_custom, bias_custom, sv_features, axis=axis1, title="Custom SVM Decision Boundary")
    
    visualize_decision_boundary(features, labels, None, None, classifier.support_vectors_, classifier=classifier, axis=axis2, title="Scikit-learn SVM Decision Boundary")
    
    plt.legend()
    plt.show()

compare_svm_with_sklearn(features, labels, weights, bias)

print('Executing Part 2')

X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)
y_moons = np.where(y_moons == 0, -1, 1)  
X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(X_moons, y_moons, test_size=0.3, random_state=42)

def gaussian_kernel_function(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

def build_kernel_matrix(X, sigma=1.0):
    num_samples = X.shape[0]
    kernel_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            kernel_matrix[i, j] = gaussian_kernel_function(X[i], X[j], sigma)
    return kernel_matrix

def fit_dual_svm(X, y, C=1.0, sigma=1.0):
    num_samples, num_features = X.shape
    K = build_kernel_matrix(X, sigma)
    
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(num_samples))
    G = matrix(np.vstack((-np.eye(num_samples), np.eye(num_samples))))
    h = matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * C)))
    A = matrix(y.astype(float), (1, num_samples))
    b = matrix(0.0)
    
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    
    alphas = np.array(solution['x']).flatten()
    
    sv_condition = alphas > 1e-5
    sv_indices = np.where(sv_condition)[0]
    return alphas, sv_indices

alphas_dual, sv_indices_dual = fit_dual_svm(X_train_moons, y_train_moons, C=1.0, sigma=1.0)
support_vectors_dual = X_train_moons[sv_indices_dual]
support_vector_labels_dual = y_train_moons[sv_indices_dual]

def predict_with_dual_svm(X_train, y_train, alphas, X_test, sigma=1.0):
    predictions = []
    for x in X_test:
        pred = np.sum(
            alphas[sv_indices_dual] * support_vector_labels_dual *
            np.array([gaussian_kernel_function(x, sv, sigma) for sv in support_vectors_dual])
        )
        predictions.append(np.sign(pred))
    return np.array(predictions)

y_pred_dual = predict_with_dual_svm(X_train_moons, y_train_moons, alphas_dual, X_test_moons, sigma=1.0)

accuracy_dual = accuracy_score(y_test_moons, y_pred_dual)

classifier_sklearn = SVC(kernel='rbf', gamma='scale')
classifier_sklearn.fit(X_train_moons, y_train_moons)

y_pred_sklearn = classifier_sklearn.predict(X_test_moons)

accuracy_sklearn = accuracy_score(y_test_moons, y_pred_sklearn)

def visualize_decision_boundary_model(X, y, model, title):
    h = .02  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap=plt.cm.binary, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
visualize_decision_boundary_model(X_train_moons, y_train_moons, classifier_sklearn, "Custom Dual SVM Decision Boundary")

plt.subplot(1, 2, 2)
visualize_decision_boundary_model(X_train_moons, y_train_moons, classifier_sklearn, "Scikit-learn SVM Decision Boundary")

plt.tight_layout()
plt.show()

print('Executing Part 3')

np.random.seed(42)
num_samples = 100
X_data = np.sort(2 * np.pi * np.random.rand(num_samples, 1), axis=0)
y_data = np.sin(X_data).ravel() + np.random.normal(0, 0.1, num_samples)

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

poly_degrees = np.arange(1, 11)
train_errors = []
test_errors = []

for degree in poly_degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train_data)
    X_test_poly = poly_features.transform(X_test_data)
    
    regression_model = LinearRegression()
    regression_model.fit(X_train_poly, y_train_data)
    
    y_train_pred = regression_model.predict(X_train_poly)
    y_test_pred = regression_model.predict(X_test_poly)
    
    train_errors.append(mean_squared_error(y_train_data, y_train_pred))
    test_errors.append(mean_squared_error(y_test_data, y_test_pred))

plt.figure(figsize=(8, 6))
plt.plot(poly_degrees, train_errors, label='Training Error', marker='o')
plt.plot(poly_degrees, test_errors, label='Testing Error', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Exploring Bias-Variance Tradeoff in Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
