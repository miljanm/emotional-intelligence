__author__ = 'matteo'

import numpy as np

a = np.zeros((2,2))
b = np.ones((2,2))

c = np.concatenate((a,b), axis=0)

print c