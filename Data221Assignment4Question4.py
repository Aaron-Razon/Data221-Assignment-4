# Q4 - Neural Network for Binary Classification

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

# === Set Random Seeds for Reproducibility ===
np.random.seed(42)
tf.random.set_seed(42)

# === Load the Dataset ===
breast_cancer_dataset = load_breast_cancer()

# === Create the Feature Matrix X and Target Vector y ===
feature_matrix_X = breast_cancer_dataset.data
target_vector_y = breast_cancer_dataset.target

# === Split the Data Into Training and Testing Sets ===
training_feature_matrix_X, testing_feature_matrix_X, training_target_vector_y, testing_target_vector_y = train_test_split(
    feature_matrix_X,
    target_vector_y,
    test_size=0.20,
    stratify=target_vector_y,
    random_state=42
)

# === Standardize the Input Features ===
feature_scaler = StandardScaler()

standardized_training_feature_matrix_X = feature_scaler.fit_transform(training_feature_matrix_X)
standardized_testing_feature_matrix_X = feature_scaler.transform(testing_feature_matrix_X)

# === Build the Neural Network Model ===
binary_classification_neural_network_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(standardized_training_feature_matrix_X.shape[1],)),
    tf.keras.layers.Dense(1, activation="sigmoid")]
)

# === Compile the Model ===
binary_classification_neural_network_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# === Train the Model ===
binary_classification_neural_network_model.fit(
    standardized_training_feature_matrix_X,
    training_target_vector_y,
    epochs=30,
    batch_size=32,
    verbose=0
)

# === Evaluate the Model on the Training Set ===
training_loss, training_accuracy = binary_classification_neural_network_model.evaluate(
    standardized_training_feature_matrix_X,
    training_target_vector_y,
    verbose=0
)

# === Evaluate the Model on the Test Set ===
testing_loss, testing_accuracy = binary_classification_neural_network_model.evaluate(
    standardized_testing_feature_matrix_X,
    testing_target_vector_y,
    verbose=0
)

# === Print the Results ===
print("Training Accuracy:", training_accuracy)
print("Test Accuracy:", testing_accuracy)