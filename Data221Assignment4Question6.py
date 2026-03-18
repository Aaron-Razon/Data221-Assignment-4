# Q6 - Convolutional Neural Network with Fashion MNIST

from tensorflow.keras.datasets import fashion_mnist

# === Load the Fashion MNIST Dataset ===
(training_images_X, training_labels_y), (testing_images_X, testing_labels_y) = fashion_mnist.load_data()

# === Normalize the Pixel Values to the Range [0, 1] ===
training_images_X = training_images_X / 255.0
testing_images_X = testing_images_X / 255.0

# === Reshape the Images to Include a Channel Dimension ===
training_images_X = training_images_X.reshape(60000, 28, 28, 1)
testing_images_X = testing_images_X.reshape(10000, 28, 28, 1)