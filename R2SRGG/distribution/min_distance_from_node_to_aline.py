import numpy as np
import matplotlib.pyplot as plt

from R2SRGG.R2SRGG import dist_to_geodesic_R2, R2SRGG
from main import find_k_connected_node_pairs


def generate_random_line():
    """Generate a random line by two points on the square boundary."""
    x = np.random.rand(2,2)
    x1, y1 = x[0]
    x2, y2 = x[1]
    return np.array([x1, y1]), np.array([x2, y2])


def simulate(N=100, num_trials=1000,h=1, plot_example=False):
    """Simulate the experiment and compute expected minimal distance."""
    min_distances = []

    for _ in range(num_trials):
        # Step a: random line
        p1, p2 = generate_random_line()

        # Step b: random points in the square
        points = np.random.rand(N, 2)

        # Step c: distances
        distances = [dist_to_geodesic_R2(pt[0],pt[1], p1[0],p1[1], p2[0],p2[1])[0] for pt in points]
        if h == 1:
            min_dist = min(distances)
        else:
            min_dist_vec = sorted(distances)[:h]
            min_dist=np.mean(min_dist_vec)
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
    # ave_deviation_dict,std_deviation_dict,kvec_dict = load_hop_count()
    # print(ave_deviation_dict)

    y_simu =[]

    SP_nodenum_vec = [int(i) for i in
                      [np.float64(8.235555555555555), np.float64(13.7447), np.float64(20.7283), np.float64(39.0912),
                       np.float64(56.7094),
                       np.float64(91.4394), np.float64(135.004)]]
    hop_vec = [int(i) for i in
               [np.float64(5.3115151515151515), np.float64(7.6669), np.float64(11.4205), np.float64(15.891),
                np.float64(22.8347), np.float64(32.7618), np.float64(47.5337)]]
    # for N in [46, 100, 215, 464, 1000, 2154, 4642, 10000]:
    count = 0
    for N in [100, 215, 464, 1000, 2154, 4642, 10000]:
        # ave_h = SP_nodenum_vec[count]
        ave_h = hop_vec[count]
        # ave_h = ave_deviation_dict[N][1]
        expected_distance, _ = simulate(N, num_trials=1000, h=int(ave_h), plot_example=False)
        y_simu.append(expected_distance)

        count = count + 1
    print(y_simu)



def load_data(N, kvec, beta, filefoldername):
    exemptionlist = []
    for N in [N]:
        ave_hop_vec = []
        std_hop_vec = []
        real_ave_degree_vec = []
        # sp_nodenum_vec = []
        for beta in [beta]:
            for ED in kvec:
                for ExternalSimutime in [0]:
                    try:
                        ave_deviation_name = filefoldername+ "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)
                        ave_hop_vec.append(np.mean(ave_deviation_for_a_para_comb))
                        std_hop_vec.append(np.std(ave_deviation_for_a_para_comb))

                    except FileNotFoundError:
                        ave_deviation_name = filefoldername + "SPhopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb = np.loadtxt(ave_deviation_name)
                        ave_hop_vec.append(np.mean(ave_deviation_for_a_para_comb))
                        std_hop_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    return kvec, real_ave_degree_vec, ave_hop_vec, std_hop_vec



def load_hop_count():
    Nvec = [46, 100, 215, 464, 1000, 2154, 4642, 10000]
    # Nvec = [46, 100, 215, 464, 1000, 2154]
    beta = 1024
    # kvec = [8,10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120,155]
    kvec = [8, 10, 13, 17, 22, 28, 36, 46, 58, 74, 94, 120]
    ave_deviation_dict ={}
    std_deviation_dict ={}
    kvec_dict={}
    for N in Nvec:
        if N < 400:
            filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\smallnetwork\\"
            kvec, _, ave_hop_vec, std_deviation_vec = load_data(N, kvec, beta, filefolder_name)
            # real_ave_degree_dict[N] = real_ave_degree_vec
            ave_deviation_dict[N] = ave_hop_vec
            std_deviation_dict[N] = std_deviation_vec
            kvec_dict[N] = kvec
        else:
            filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\"
            kvec, real_ave_degree_vec, ave_hop_vec, std_deviation_vec = load_data(N, kvec, beta, filefolder_name)
            # real_ave_degree_dict[N] = real_ave_degree_vec
            kvec_dict[N] = kvec
            ave_deviation_dict[N] = ave_hop_vec
            std_deviation_dict[N] = std_deviation_vec

    return ave_deviation_dict,std_deviation_dict,kvec_dict


if __name__ == '__main__':
    # Example usage:
    # simulate(N=100, num_trials=1000, plot_example=True)
    simu_diff_N()

