# Q5 - Model Evaluation and Comparison

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# === Set Random Seeds for Reproducibility ===
np.random.seed(42)
tf.random.set_random_seed(42)

# === Load the Dataset ===
breast_cancer_dataset = load_breast_cancer()

# === Create Feature Matrix X and Target Vector y ===
feature_matrix_X = breast_cancer_dataset.data
target_vector_y = breast_cancer_dataset.target

# === Split the Data into Training and Testing Sets ===
training_feature_matrix_X, test_feature_matrix_X, training_target_vector_y, test_target_vector_y = train_test_split(
    feature_matrix_X,
    target_vector_y,
    test_size=0.2,
    random_state=42,
    stratify=target_vector_y
)

