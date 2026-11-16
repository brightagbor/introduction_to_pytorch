import numpy
import torch
import torch.nn as nn


# What the model below does
# 1. Start  with random values (weight and bias)
# 2. Look at training data and adjust teh random values to better represent
#    (or get closer to) the ideal values (the weight and bias values we used to creat the data.)

# This is done using two main algorithms
# 1. Gradient descent
# 2. Backpropagation


# NOTE, in a more suitable dataset, we won't necessarily be defining the
# parameters (weight and bias) instead another module in nn will define the parameters.

# Create a linear regression model class
class LinearRegressionModel(nn.Model):
    def __init__(self):
        super().__init__()

        # Initialise model parameters
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch,
                                 requires_grad=True,
                                 dtype=torch.float)
    
    # Override the forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- `x` is the input data
        #Below is the linear regression formula
        return self.weight  * x + self.bias