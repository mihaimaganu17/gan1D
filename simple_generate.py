# Generating random sample from the `square(x)` space
import numpy as np
from matplotlib import pyplot

def generate_samples(n=100):
    # Generate random input in [-0.5, -0.5] of shape `n`
    x1 = np.random.rand(n) - 0.5
    # Generate outputs of `square(x)`
    x2 = x1*x1
    # Reshape them over rows
    x = np.vstack((x1, x2))
    return x.T


# Generates real samples for `square(x)` function
# Returns the generated samples and the labels(1=real)
def generate_real_samples(n):
    # Generate x points
    x = generate_samples(n)
    # Generate labels
    y = np.ones((n, 1))
    return x, y


# Generates fake samples for `square(x)` function
# Returns the generated samples and the labels(0=fake)
def generate_fake_samples(n):
    # Generate inputs in range [-1, 1]
    x1 = np.random.rand(n) * 2 - 1.0
    # Generate outputs in range [-1, 1]
    x2 = np.random.rand(n) * 2 - 1.0
    # Stack array
    x = np.vstack((x1, x2))
    # Return as an [2; n] shape array
    x = x.T
    # Generate labels
    y = np.zeros((n, 1))
    return x, y


if __name__ == "__main__":
    # Generate samples
    data = generate_samples()
    # Plot samples
    pyplot.scatter(data[:, 0], data[:, 1])
    pyplot.show()

