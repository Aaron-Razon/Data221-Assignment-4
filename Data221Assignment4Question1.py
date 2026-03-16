# Q1 - Dataset Exploration and Understanding

from sklearn.datasets import load_breast_cancer
import pandas as pd

# === Load the Dataset ===
breast_cancer_dataset = load_breast_cancer()

# === Create Feature Matrix X and Target Vector y ===
feature_matrix_X = breast_cancer_dataset.data
target_vector_y = breast_cancer_dataset.target

# === Report the Shapes of X and y ===
print("Shape of Feature Matrix: ", feature_matrix_X.shape)
print("Shape of Target Vector: ", target_vector_y.shape)

# === Count How Many Samples are in Each Class ===
target_class_counts = pd.Series(target_vector_y).value_counts()

# === Output Class Counts ===
print("\nClass Counts: ")
for class_name, class_count in target_class_counts.items():
    class_name = breast_cancer_dataset.target_names[class_name]
    print(f"{class_name}: {class_count}")