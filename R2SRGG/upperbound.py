# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 6-9-2024
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# Function to calculate the probability function from the provided equation
def probability_function(d, N, M, t, k):
    r = 0.001  # Assuming r = 1 for simplicity as it's not provided
    result = 0
    hop = np.floor(k/2)
    for n in range(hop):
        term = (t * N / M)*(hop*r - d) * np.exp(-(t * N / M) * (r - d)) / factorial(n)
        result += term
    return result





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    N = 10000
    M = 1
    t = 0.5
    k = 4
    # Values of d to plot
    d_values = np.linspace(0, 0.01, 100)

    # Calculate the corresponding probabilities
    probabilities = [probability_function(d, N, M, t, k) for d in d_values]

    # Plotting the function
    plt.plot(d_values, probabilities, label="Probability Function")
    plt.title("Probability Function vs d")
    plt.xlabel("d")
    plt.yscale("log")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.show()
