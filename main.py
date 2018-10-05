from perceptron import *
from plots import *
import random
import numpy as np

alpha = 0.4
x_bias = 1
x_all = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_all = np.array([0, 0, 0, 1])

w = np.array([])
for i in range(n+1):
    w = np.append(w, [random.uniform(0, 1)*2-1])


weights = train(x_all, y_all, w, alpha, x_bias)

x_test1 = np.array([[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]])
x_test2 = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])

print('\nTest on default data:')
test_model(x_all, y_all, weights, x_bias)

print('\nTest on modified data (0.05 and 0.95):')
test_model(x_test1, y_all, weights, x_bias)

print('\nTest on modified data (0.2 and 0.8):')
test_model(x_test2, y_all, weights, x_bias)


''' DRAWING '''
draw_plot([x_all, x_test1, x_test2], weights, y_all)
