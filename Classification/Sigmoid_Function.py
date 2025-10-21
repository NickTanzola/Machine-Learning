#here is the implementation of the sigmoid function for logistic regression
import numpy as np
import math
#z is the value of the model's prediction
def sigmoid_function(z):
  return 1 / (1 + math.exp(-z))
