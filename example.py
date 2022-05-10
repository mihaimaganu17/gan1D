# Module that demontrates a simple square function
from matplotlib import pyplot
import numpy as np

# Computes the square of the input `x`
def square(x):
    return x * x

# Define inputs
inputs = [i for i in np.arange(-0.5, 0.6, 0.1)]

# Compute outputs
outputs = [square(x) for x in inputs]

# Plot the result -> a U-shape plot
pyplot.plot(inputs, outputs)
pyplot.show()

