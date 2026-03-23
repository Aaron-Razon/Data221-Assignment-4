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

# This dataset is slightly imbalanced, but not severely imbalanced.
# There are more benign samples than malignant samples.

# Class balance is important in classification because if one class
# appears much more often than the other, a model may become biased
# toward the majority class. This can make the model look accurate
# overall while still performing poorly on the minority class.