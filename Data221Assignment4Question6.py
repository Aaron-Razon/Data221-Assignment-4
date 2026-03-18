# Q6 - Convolutional Neural Network with Fashion MNIST

from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf

# === Load the Fashion MNIST Dataset ===
(training_images_X, training_labels_y), (testing_images_X, testing_labels_y) = fashion_mnist.load_data()

# === Normalize the Pixel Values to the Range [0, 1] ===
training_images_X = training_images_X / 255.0
testing_images_X = testing_images_X / 255.0

# === Reshape the Images to Include a Channel Dimension ===
training_images_X = training_images_X.reshape(60000, 28, 28, 1)
testing_images_X = testing_images_X.reshape(10000, 28, 28, 1)

# === Build the CNN Model ===
cnn_model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

# === Compile the Model ===
cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# === Train the Model ===
cnn_model.fit(
    training_images_X,
    training_labels_y,
    epochs=15
)

# === Evaluate the Model on the Test Set ===
test_loss, test_accuracy = cnn_model.evaluate(testing_images_X, testing_labels_y)

# === Report the Test Accuracy ===
print("Test Accuracy:", test_accuracy)