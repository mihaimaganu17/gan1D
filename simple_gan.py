import numpy as np
from matplotlib import pyplot
from tensorflow.keras import models, layers, utils

# Modules defined in this package
from simple_discriminator import discriminator_model
from simple_generator import generator_model, generate_latent_space, generate_fake_samples
from simple_generate import generate_real_samples


# Create a new GAN logic model based on a `generator` and `discriminator` model
def gan(generator_model, discriminator_model):
    # Make the weights in the discriminator not trainable when it is part of the GAN model.
    # This is to make sure that we train the generator for generating better fake samples
    discriminator_model.trainable = False

    # Define a new sequential model
    model = models.Sequential()
    # Add the generator
    model.add(generator_model)
    # Add the discriminator
    model.add(discriminator_model)
    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam")

    return model


# Trains a GAN model over the desired number of epochs `n_epochs` and with a batch of size
# `n_batch`
def train_gan(gan_model, latent_space_size, n_epochs=1000, n_batch=150):
    # For each epoch
    for epoch in range(n_epochs):
        # Prepare latent space as input for the generator
        x_latent_space = generate_latent_space(latent_space_size, n_batch)
        # Add `real` labels for the fake samples
        y_labels = np.ones((n_batch, 1))
        # Train GAN by updating the generator using discriminator's error
        gan_model.train_on_batch(x_latent_points, y_labels)


# Train the gan correctly
def train(
    generator_model,
    discriminator_model,
    gan_model,
    latent_space_size,
    n_epochs=10000,
    n_batch=128,
    n_eval=2000
):
    # Determine the half size for one batch
    half_batch_size = int(n_batch / 2)
    # For each epoch
    for epoch in range(n_epochs):
        # Prepare real samples
        x_real, y_real = generate_real_samples(half_batch_size)
        # Generate fake samples
        x_fake, y_fake = generate_fake_samples(generator_model, latent_space_size, half_batch_size)

        discriminator_model.trainable = True
        # Train the discriminator for both batches
        discriminator_model.train_on_batch(x_real, y_real)
        discriminator_model.train_on_batch(x_fake, y_fake)

        discriminator_model.trainable = False

        # Generate latent space
        x_gan = generate_latent_space(latent_space_size, n_batch)
        # Create `real` labels for fake examples
        y_gan = np.ones((n_batch, 1))

        gan_model.train_on_batch(x_gan, y_gan)

        if (epoch + 1) % n_eval == 0:
            summarize_performance(epoch, generator_model, discriminator_model, latent_space_size)


def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # Get real samples
    x_real, y_real = generate_real_samples(n)

    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)

    # Get fake samples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)

    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

    print(epoch, acc_real, acc_fake)

    # Scatter plot real and fake data points
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color="red")
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color="blue")
    pyplot.show()


# Define a size for the latent space
latent_space_size = 5
# Create a discriminator model
discriminator = discriminator_model()
# Create a generator model
generator = generator_model(latent_space_size)
# Create a GAN model based on the 2 above
gan = gan(generator, discriminator)

train(generator, discriminator, gan, latent_space_size)
