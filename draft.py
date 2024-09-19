import numpy as np

# a = np.array([])
a = {1:{2:[1,2,3]}}
b = {2:[4,5,6]}
try:
    a[1][2] = a[1][2] + list([1,3])
except:
    a[1][2] = list([1,3])


print(a)