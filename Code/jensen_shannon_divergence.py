import scipy
import numpy as np
from scipy.spatial import distance

#this is just a code we were trying on random data
distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)

a = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]])
b = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]])
print(distance.jensenshannon(a, b, axis=0))
#print(distance.jensenshannon(a, b, axis=1))
#then maybe a for loop
#we need to use Panda to map the features with the IPs 
