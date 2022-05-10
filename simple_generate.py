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


# Generate samples
data = generate_samples()
# Plot samples
pyplot.scatter(data[:, 0], data[:, 1])
pyplot.show()


