import numpy as np

A = np.matrix([[0, 2, 4], [2, 4, 2], [3, 3, 1]])
b = np.array([[-2], [-2], [-4]])
c = np.array([[1], [1], [1]])

print('a. A^{-1} = \n', A.getI(), '\n')
print('b. A^{-1} * b = \n', A.getI() * b, '\n')
print('b. A * c = \n', A * c)
