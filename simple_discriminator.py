from matplotlib import pyplot
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils

# Our modules
from simple_generate import generate_real_samples, generate_fake_samples


# Function that defines and returns a discriminator model
def discriminator_model(n_inputs=2):
    # Define a Sequential model
    model = models.Sequential()
    # Add a dense layer with 25 neurons
    model.add(layers.Dense(
        25,
        activation="relu",
        kernel_initializer="he_uniform",
        input_dim=n_inputs
    ))
    # Add a dense layer as output layer
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam",
            metrics=["accuracy"])

    # Return the created model
    return model


# Function to train a discriminator model
def train_discriminator(model, n_epochs=1000, n_batch=128):
    # Compute length of a half batch
    n_half_batch = int(n_batch / 2)

    # For each epoch
    for epoch in range(n_epochs):
        # Generate half real examples
        x_real, y_real = generate_real_samples(n_half_batch)
        # Train on real batch
        model.train_on_batch(x_real, y_real)

        # Generate half fake examples
        x_fake, y_fake = generate_fake_samples(n_half_batch)
        # Train on fake batch
        model.train_on_batch(x_fake, y_fake)

        _, acc_real = model.evaluate(x_real, y_real, verbose=0)
        _, acc_fake = model.evaluate(x_fake, y_fake, verbose=0)

        print(epoch, acc_real, acc_fake)


if __name__ == "__main__":
    # Define the discriminator model
    model = discriminator_model()
    # fit the model
    train_discriminator(model)

