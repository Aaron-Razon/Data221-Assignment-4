# Q3 - Controlling Tree Complexity and Interpretability

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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