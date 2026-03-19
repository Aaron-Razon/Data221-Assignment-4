# Q7 - CNN Error Analysis and Misclassification Study

from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Load the Fashion MNIST Dataset ===
(training_images_X, training_labels_y), (testing_images_X, testing_labels_y) = fashion_mnist.load_data()

# === Normalize the Pixel Values to the Range [0, 1] ===
training_images_X = training_images_X / 255.0
testing_images_X = testing_images_X / 255.0

# === Reshape the Images to Include a Channel Dimension ===
training_images_X = training_images_X.reshape(60000, 28, 28, 1)
testing_images_X = testing_images_X.reshape(10000, 28, 28, 1)

# === Build the CNN Model ===
fashion_mnist_cnn_model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

# === Compile the Model ===
fashion_mnist_cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# === Train the Model ===
fashion_mnist_cnn_model.fit(
    training_images_X,
    training_labels_y,
    epochs=15,
    batch_size=32,
    verbose=1
)

# === Generate Predictions on the Test Set ===
prediction_probabilities = fashion_mnist_cnn_model.predict(testing_images_X, verbose=0)
predicted_labels_y = np.argmax(prediction_probabilities, axis=1)

# === Compute the Confusion Matrix ===
cnn_confusion_matrix = confusion_matrix(testing_labels_y, predicted_labels_y)

print("CNN Confusion Matrix:")
print(cnn_confusion_matrix)


# === Display the Confusion Matrix ===
fashion_mnist_label_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

ConfusionMatrixDisplay(
    confusion_matrix=cnn_confusion_matrix,
    display_labels=fashion_mnist_label_names
).plot(xticks_rotation=45)

plt.title("CNN Confusion Matrix")
plt.show()

# === Find Misclassified Images ===
misclassified_image_indexes = np.where(predicted_labels_y != testing_labels_y)[0]

# === Display Three Misclassified Images ===
plt.figure(figsize=(10, 4))

for image_position in range(3):
    misclassified_index = misclassified_image_indexes[image_position]

    plt.subplot(1, 3, image_position + 1)
    plt.imshow(testing_images_X[misclassified_index].reshape(28, 28), cmap="gray")
    plt.title(
        f"True: {fashion_mnist_label_names[testing_labels_y[misclassified_index]]}\n"
        f"Predicted: {fashion_mnist_label_names[predicted_labels_y[misclassified_index]]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()