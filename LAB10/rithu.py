import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

def main():
    # Clustering
    np.random.seed(0)
    a = np.random.randint(1, 31, (25, 2))
    b = np.random.randint(45, 76, (25, 2))
    c = np.random.randint(150, 201, (25, 2))
    dataset = np.concatenate((a, b, c), axis=0)

    silhouette_scores, k_values = [], range(2, 10)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(dataset)
        score = silhouette_score(dataset, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"k = {k}, Silhouette Score = {score:.4f}")
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters (k) according to Silhouette score: {optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(dataset)
    labels = kmeans.labels_

    plt.figure(figsize=(10, 7))
    for i in range(optimal_k):
        plt.scatter(dataset[labels == i][:, 0], dataset[labels == i][:, 1], label=f'Cluster {i+1}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('K-Means Clustering with Optimal k')
    plt.legend()
    plt.show()

    # MNIST Classification
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
    x_train_full, x_test = x_train_full / 255.0, x_test / 255.0
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.15, random_state=42)
    y_train, y_val, y_test = to_categorical(y_train, 10), to_categorical(y_val, 10), to_categorical(y_test, 10)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    learning_rate, batch_size, epochs = 0.001, 128, 20
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nModel Hyperparameters:")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")

if __name__ == "__main__":
    main()