import numpy as np

data = np.array([
    [1, 2, 3],
    [1, 2, 5],
    [1, 3, 7],
    [2, 2, 6],
    [1, 2, 8],
    [2, 2, 9]
])

# 筛选出A == 1 且 B == 2 的行
x_vec = set([1,1,2,3])
y_vec = set([3,2,1])
for x in x_vec:
    for y in y_vec:
        indices = np.where((data[:, 0] == x) & (data[:, 1] == y))[0]
        print(indices)
        for i in indices:
            print(i)