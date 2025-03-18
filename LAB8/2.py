import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from cvxopt import matrix, solvers

# Step 1: Generate the two moons dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Custom RBF Kernel function
def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

# Step 3: Create the kernel matrix for the training set
def create_kernel_matrix(X, sigma=1.0):
    m = X.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = gaussian_kernel(X[i], X[j], sigma)
    return K

# Step 4: SVM Dual Form Optimization (using CVXOPT solver)
def fit_svm_dual(X, y, C=1.0, sigma=1.0):
    m, n = X.shape
    K = create_kernel_matrix(X, sigma)
    
    # Construct matrices for the quadratic programming problem
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(m))
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(y.astype(float), (1, m))
    b = matrix(0.0)
    
    # Solve the quadratic programming problem using cvxopt
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    
    # Extract Lagrange multipliers
    alphas = np.array(solution['x']).flatten()
    
    # Support vectors are the ones with non-zero Lagrange multipliers
    support_vectors = alphas > 1e-5
    sv_indices = np.where(support_vectors)[0]
    return alphas, sv_indices

# Step 5: Train the SVM model using the dual form
alphas, sv_indices = fit_svm_dual(X_train, y_train, C=1.0, sigma=1.0)
support_vectors = X_train[sv_indices]
support_vector_labels = y_train[sv_indices]

# Step 6: Predict using the Gaussian kernel and support vectors
def predict_svm_dual(X_train, y_train, alphas, X_test, sigma=1.0):
    y_pred = []
    for x in X_test:
        prediction = np.sum(
            alphas[sv_indices] * support_vector_labels *
            np.array([gaussian_kernel(x, sv, sigma) for sv in support_vectors])
        )
        y_pred.append(np.sign(prediction))
    return np.array(y_pred)

# Step 7: Predict the test data
y_pred_dual = predict_svm_dual(X_train, y_train, alphas, X_test, sigma=1.0)

# Step 8: Print accuracy for the custom dual SVM
accuracy_dual = accuracy_score(y_test, y_pred_dual)
print(f"Accuracy (Custom Dual SVM): {accuracy_dual * 100:.2f}%")

# Step 9: Train an SVM using scikit-learn's SVC with RBF kernel
clf = SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

# Step 10: Predictions using scikit-learn
y_pred_sklearn = clf.predict(X_test)

# Step 11: Print accuracy for scikit-learn SVM
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Accuracy (Scikit-learn SVM): {accuracy_sklearn * 100:.2f}%")

# Step 12: Visualization function for decision boundaries
def plot_decision_boundary(X, y, model, title):
    h = .02  # step size in the mesh
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

# Step 13: Plot the decision boundaries for both SVMs in a single window
plt.figure(figsize=(12, 5))

# Custom Dual SVM Decision Boundary
plt.subplot(1, 2, 1)
plot_decision_boundary(X_train, y_train, clf, "Custom Dual SVM Decision Boundary")

# Scikit-learn SVM Decision Boundary
plt.subplot(1, 2, 2)
plot_decision_boundary(X_train, y_train, clf, "Scikit-learn SVM Decision Boundary")

plt.tight_layout()
plt.show()
