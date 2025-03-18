import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
import pandas as pd

def perform_clustering():
    a = np.random.randint(1, 31, size=(25, 2))
    b = np.random.randint(45, 76, size=(25, 2))
    c = np.random.randint(150, 201, size=(25, 2))
    dataset = np.concatenate((a, b, c), axis=0)

    silhouette_scores = []
    k_values = range(2, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(dataset)
        score = silhouette_score(dataset, labels)
        silhouette_scores.append(score)
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters (k): {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(dataset)
    centroids = kmeans.cluster_centers_
    sample_silhouette_values = silhouette_samples(dataset, labels)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    base_colors = plt.colormaps["tab10"](np.linspace(0, 1, optimal_k))
    colors = [base_colors[i] for i in range(optimal_k)]
    for i in range(optimal_k):
        cluster_points = dataset[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[i], label=f'Cluster {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, color='red', marker='X', label='Centroids')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title(f"K-Means Clustering with k={optimal_k}")

    plt.subplot(1, 2, 2)
    plt.bar(k_values, silhouette_scores, color="skyblue")
    plt.axvline(x=optimal_k, color="red", linestyle="--")
    plt.title("Silhouette Scores for Different k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.show()

    distances = [np.linalg.norm(dataset[labels == i] - centroids[i], axis=1).mean() for i in range(optimal_k)]
    print("Average Distance from Centroids per Cluster:", distances)

def perform_classification():
    layer_sizes = [512, 256, 128, 64, 32, 16]
    activation_function = 'relu'
    dropout_rates = [0.3, 0.3, 0.25, 0.25, 0.2, 0.2]
    learning_rate = 0.001
    batch_size = 64
    epochs = 25
    patience = 10

    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_full, x_test = x_train_full / 255.0, x_test / 255.0
    x_train_full = x_train_full.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train_full = to_categorical(y_train_full, 10)
    y_test = to_categorical(y_test, 10)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.15, random_state=42)

    def scheduler(epoch, lr):
        return lr * 0.95 if epoch > 10 else lr  

    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(layer_sizes[0], activation=activation_function), BatchNormalization(), Dropout(dropout_rates[0]),
        Dense(layer_sizes[1], activation=activation_function), BatchNormalization(), Dropout(dropout_rates[1]),
        Dense(layer_sizes[2], activation=activation_function), BatchNormalization(), Dropout(dropout_rates[2]),
        Dense(layer_sizes[3], activation=activation_function), BatchNormalization(), Dropout(dropout_rates[3]),
        Dense(layer_sizes[4], activation=activation_function), BatchNormalization(), Dropout(dropout_rates[4]),
        Dense(layer_sizes[5], activation=activation_function), BatchNormalization(), Dropout(dropout_rates[5]),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(scheduler)
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, 
                        callbacks=[early_stopping, lr_scheduler], verbose=1)
    
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    hyperparameters = {
        "Layer Sizes": layer_sizes,
        "Activation Function": activation_function,
        "Dropout Rates": dropout_rates,
        "Learning Rate": learning_rate,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Early Stopping Patience": patience
    }
    print("\nHyperparameters Used:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    

def main():
    print("Performing K-means Clustering:")
    perform_clustering()
    print("\nPerforming MNIST Classification with MLP:")
    perform_classification()

if __name__ == "__main__":
    main()
