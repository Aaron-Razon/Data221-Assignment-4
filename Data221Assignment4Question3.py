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

# === Create the Constrained Decision Tree Model ===
# max_depth=3 is used to limit how deep the tree can grow
constrained_decision_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

# === Train the Model ===
constrained_decision_tree_model.fit(training_feature_matrix_X, training_target_vector_y)

# === Make Predictions on the Training Set and Test Set ===
training_predictions_y = constrained_decision_tree_model.predict(training_feature_matrix_X)
testing_predictions_y = constrained_decision_tree_model.predict(testing_feature_matrix_X)

# === Calculate Accuracy for Both Sets ===
training_accuracy = accuracy_score(training_target_vector_y, training_predictions_y)
testing_accuracy = accuracy_score(testing_target_vector_y, testing_predictions_y)

# === Print the Accuracy Results ===
print("Training Accuracy:", training_accuracy)
print("Test Accuracy:", testing_accuracy)

# === Find the Top Five Most Important Features ===
feature_importance_pairs = list(
    zip(breast_cancer_dataset.feature_names, constrained_decision_tree_model.feature_importances_)
)

sorted_feature_importance_pairs = sorted(
    feature_importance_pairs,
    key=lambda feature_pair: feature_pair[1],
    reverse=True
)

top_five_feature_importance_pairs = sorted_feature_importance_pairs[:5]

# === Print the Top Five Most Important Features ===
print("\nTop Five Most Important Features:")
for feature_name, importance_score in top_five_feature_importance_pairs:
    print(f"{feature_name}: {importance_score:.4f}")