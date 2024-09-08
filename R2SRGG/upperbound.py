# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 6-9-2024
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import math

# Function to calculate the probability function from the provided equation
def probability_function(d, N, M, t, k, r):
    result = 0
    hop = int(math.floor(k/2))
    for n in range(hop):
        a = (t * N / M) * (hop * r - d)
        print(a)
        numerator2 = math.exp(-a)
        print(numerator2)
        numerator1 = a**hop
        print(numerator1)
        # term = ()**hop * math.exp(-(t * N / M) * (hop*r - d)) / factorial(n)
        # result += term

    return result





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    N = 10000
    M = 1
    ED = 5
    r = ED * M / ((N - 1) * math.pi)
    t = 0.5
    k = math.ceil(t/r)+10

    # Values of d to plot
    hop = math.floor(k / 2)
    xiebian = hop * r
    xiebian = xiebian**2
    d_max = math.sqrt(( hop* r)**2 - (t / 2)**2)
    d_values = np.linspace(0, d_max, 10)

    assert (k*r>t)

    # Calculate the corresponding probabilities
    probabilities = [probability_function(d, N, M, t, k, r) for d in [0.01]]
    # probabilities = [probability_function(d, N, M, t, k, r) for d in d_values]
    #
    # # Plotting the function
    # plt.plot(d_values, probabilities, label="Probability Function")
    # plt.title("Probability Function vs d")
    # plt.xlabel("d")
    # plt.yscale("log")
    # plt.ylabel("Probability")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
