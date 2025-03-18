import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load the MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 2: Preprocess the data
x_train_full = x_train_full / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test / 255.0

# Reshape for the MLP input (Flattening will be handled in the model)
x_train_full = x_train_full.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_full = to_categorical(y_train_full, 10)
y_test = to_categorical(y_test, 10)

# Step 3: Train-Test Split (85% Train, 15% Validation)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.15, random_state=42)

# Hyperparameters
num_layers = 6
layer_sizes = [512, 256, 128, 64, 32, 16]
activation_function = 'relu'
dropout_rates = [0.3, 0.3, 0.25, 0.25, 0.2, 0.2]
learning_rate = 0.001
batch_size = 64
epochs = 25
patience = 10


# Step 4: Build the Improved MLP Model with 6 layers
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    
    Dense(layer_sizes[0], activation=activation_function),
    BatchNormalization(),
    Dropout(dropout_rates[0]),
    
    Dense(layer_sizes[1], activation=activation_function),
    BatchNormalization(),
    Dropout(dropout_rates[1]),
    
    Dense(layer_sizes[2], activation=activation_function),
    BatchNormalization(),
    Dropout(dropout_rates[2]),
    
    Dense(layer_sizes[3], activation=activation_function),
    BatchNormalization(),
    Dropout(dropout_rates[3]),
    
    Dense(layer_sizes[4], activation=activation_function),
    BatchNormalization(),
    Dropout(dropout_rates[4]),
    
    Dense(layer_sizes[5], activation=activation_function),
    BatchNormalization(),
    Dropout(dropout_rates[5]),
    
    Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Step 5: Compile the model with a specific learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print Hyperparameters
print("Hyperparameters:")
print(f"Number of Layers: {num_layers}")
print(f"Layer Sizes: {layer_sizes}")
print(f"Activation Function: {activation_function}")
print(f"Dropout Rates: {dropout_rates}")
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Early Stopping Patience: {patience}")

# Step 6: Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

# Step 7: Train the model with increased epochs and early stopping
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)

# Step 8: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 9: Plot the training and validation accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
