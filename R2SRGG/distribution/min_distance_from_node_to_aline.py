import numpy as np
import matplotlib.pyplot as plt

from R2SRGG.R2SRGG import dist_to_geodesic_R2


def generate_random_line():
    """Generate a random line by two points on the square boundary."""
    x = np.random.rand(2,2)
    x1, y1 = x[0]
    x2, y2 = x[1]
    return np.array([x1, y1]), np.array([x2, y2])


def simulate(N=100, num_trials=1000, plot_example=False):
    """Simulate the experiment and compute expected minimal distance."""
    min_distances = []

    for _ in range(num_trials):
        # Step a: random line
        p1, p2 = generate_random_line()

        # Step b: random points in the square
        points = np.random.rand(N, 2)

        # Step c: distances
        distances = [dist_to_geodesic_R2(pt[0],pt[1], p1[0],p1[1], p2[0],p2[1])[0] for pt in points]
        min_dist = min(distances)
        min_distances.append(min_dist)

        # Optionally plot one example
        if plot_example:
            plt.figure(figsize=(5,5))
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', label='Random Line')
            plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
            plt.title(f'Min Distance: {min_dist:.4f}')
            plt.legend()
            plt.gca().set_aspect('equal')
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.show()
            plot_example = False  # Show only once

    expected_distance = np.mean(min_distances)
    print(f"Expected minimal distance over {num_trials} trials: {expected_distance:.6f}")
    return expected_distance, min_distances


def simu_diff_N():
    y_simu =[]
    for N in [46, 100, 215, 464, 1000, 2154, 4642, 10000]:
        expected_distance,_ = simulate(N,num_trials=1000, plot_example=False)
        y_simu.append(expected_distance)
    print(y_simu)



if __name__ == '__main__':
    # Example usage:
    # simulate(N=100, num_trials=1000, plot_example=True)
    simu_diff_N()

    # xMed=0.25
    # yMed=0.3
    # xSource = 0.6
    # ySource = 0.3
    # xEnd = 0.4
    # yEnd=0.4
    # a = point_to_line_distance(np.array([xMed,yMed]), np.array([xSource,ySource]), np.array([xEnd,yEnd]))
    # b = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
    # print(a)
    # print(b)