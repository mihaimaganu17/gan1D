from matplotlib import pyplot
import numpy as np
from tensorflow.keras import models, layers, utils


# Function that defines and returns a generator model
def generator_model(latent_dim, n_outputs=2):
    # Define a Sequential model
    model = models.Sequential()
    # Add a hidden layer with 15 nodes
    model.add(layers.Dense(
        5,
        activation="relu",
        kernel_initializer="he_uniform",
        input_dim=latent_dim
    ))
    # Add output layer
    model.add(layers.Dense(n_outputs, activation="linear"))
    # Return the created model
    return model


# Generates and returns point in latent space over a Gaussian distribution as input for the
# generator
def generate_latent_space(latent_dim, n):
    # Generate latent space points from a Gaussian distribution
    latent_points = np.random.rand(latent_dim, n)
    # Reshape into a batch of inputs for the network
    latent_space = latent_points.reshape(n, latent_dim)
    return latent_space


# Generate fake samples and return them
def generate_fake_samples(generator_model, latent_dim, n):
    # Generate points in the latent space
    latent_space = generate_latent_space(latent_dim, n)
    # Predict outputs
    outputs = generator_model.predict(latent_space)
    # Generate fake labels
    y_labels = np.zeros((n, 1))
    # Return the outputs
    return outputs, y_labels

# Plot data
def plot(data):
    pyplot.scatter(data[:, 0], data[:, 1])
    pyplot.show()


if __name__ == "__main__":
    latent_dim = 5
    # Get a generator model
    model = generator_model(latent_dim)
    # Generate fake samples
    fake_samples, fake_labels = generate_fake_samples(model, latent_dim, 100)
    # Plot the samples
    plot(fake_samples)
