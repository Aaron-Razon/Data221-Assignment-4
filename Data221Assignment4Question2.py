# Q2 - Decision Tree Model Using Entropy

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# === Use the Feature Matrix X and Target Vector y from Question 1 ===

# === Load the Dataset ===
breast_cancer_dataset = load_breast_cancer()

# === Create Feature Matrix X and Target Vector y ===
feature_matrix_X = breast_cancer_dataset.data
target_vector_y = breast_cancer_dataset.target

# === Split the Data Into Training and Testing Sets ===
training_feature_matrix_X, testing_feature_matrix_X, training_target_vector_y, testing_target_vector_y = train_test_split(
    feature_matrix_X,
    target_vector_y,
    test_size=0.20,
    random_state=42
)

# === Create the Decision Tree Model Using Entropy ====
entropy_decision_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# === Train the Model ===
entropy_decision_tree_model.fit(training_feature_matrix_X, training_target_vector_y)

# === Make Predictions on Both the Training Set and Test Set ===
training_predictions_y = entropy_decision_tree_model.predict(training_feature_matrix_X)
testing_predictions_y = entropy_decision_tree_model.predict(testing_feature_matrix_X)

# === Calculate Accuracy For Both Sets ===
training_accuracy = accuracy_score(training_target_vector_y, training_predictions_y)
testing_accuracy = accuracy_score(testing_target_vector_y, testing_predictions_y)

# === Print the Results ===
print("Training Accuracy:", training_accuracy)
print("Test Accuracy:", testing_accuracy)