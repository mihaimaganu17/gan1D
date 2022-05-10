from matplotlib import pyplot
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils


# Function that defines and returns a discriminator model
def discriminator_model():
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
            metrics=["acurracy"])

    # Return the created model
    return model

# Define the discriminator model
model = discriminator_model()
# summarize the model
model.summary()
# plot the model
utils.plot_model(model, to_file="discriminator_plot.png", show_shapes=True,
        show_layer_names=True)


