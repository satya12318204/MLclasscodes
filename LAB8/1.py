import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn import svm

# Step 1: Generate a Linearly Separable Dataset
def generate_linearly_separable_data(n_samples=20, separation=2.0, random_state=0):
    np.random.seed(random_state)
    
    # Generate Class -1
    mean_neg = np.array([-separation, -separation])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])  # Covariance matrix
    X_neg = np.random.multivariate_normal(mean_neg, cov, n_samples)
    y_neg = -np.ones(n_samples)
    
    # Generate Class +1
    mean_pos = np.array([separation, separation])
    X_pos = np.random.multivariate_normal(mean_pos, cov, n_samples)
    y_pos = np.ones(n_samples)
    
    # Combine the datasets
    X = np.vstack((X_neg, X_pos))
    y = np.hstack((y_neg, y_pos))
    
    return X, y

# Generate the dataset
X, y = generate_linearly_separable_data(n_samples=20, separation=3.0, random_state=42)

def svm_quadratic_programming(X, y):
    n_samples, n_features = X.shape
    
    # Compute the Gram matrix
    K = np.dot(X, X.T)
    
    # Compute the matrices for cvxopt
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n_samples))
    G = matrix(-np.eye(n_samples))
    h = matrix(np.zeros(n_samples))
    A = matrix(y.astype(float), (1, n_samples))
    b_constraint = matrix(0.0)
    
    # Suppress CVXOPT output
    solvers.options['show_progress'] = False
    
    # Solve QP problem
    solution = solvers.qp(P, q, G, h, A, b_constraint)
    
    # Lagrange multipliers
    alphas = np.ravel(solution['x'])
    
    # Support vectors have non-zero lagrange multipliers
    sv = alphas > 1e-5
    ind = np.arange(len(alphas))[sv]
    alphas_sv = alphas[sv]
    sv_X = X[sv]
    sv_y = y[sv]
    
    # Compute weight vector
    w = np.sum(alphas_sv[:, None] * sv_y[:, None] * sv_X, axis=0)
    
    # Compute bias
    b = np.mean(sv_y - np.dot(sv_X, w))
    
    return w, b, sv_X, alphas_sv

w, b, sv_X, alphas_sv = svm_quadratic_programming(X, y)

# Step 3: Plot Decision Boundaries
def plot_decision_boundary(X, y, w, b, sv_X, clf=None, ax=None, title=""):
    if ax is None:
        ax = plt.gca()
    
    # Plot all data points
    ax.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1')
    
    # Highlight support vectors
    if sv_X is not None:
        ax.scatter(sv_X[:, 0], sv_X[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    
    # Create a mesh to plot the decision boundary
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0]-1, xlim[1]+1, 500),
                         np.linspace(ylim[0]-1, ylim[1]+1, 500))
    
    # Decision function
    if clf is None:  # Custom SVM
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    else:  # Scikit-learn SVM
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)

# Step 4: Compare with scikit-learn's Built-in SVM
def compare_with_sklearn(X, y, w_custom, b_custom):
    clf = svm.SVC(kernel='linear', C=1e10)  # Large C for hard margin
    clf.fit(X, y)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Custom SVM decision boundary
    plot_decision_boundary(X, y, w_custom, b_custom, sv_X, ax=ax1, title="Custom SVM Decision Boundary")
    
    # Scikit-learn SVM decision boundary
    plot_decision_boundary(X, y, None, None, clf.support_vectors_, clf=clf, ax=ax2, title="Scikit-learn SVM Decision Boundary")
    
    plt.legend()
    plt.show()

compare_with_sklearn(X, y, w, b)
