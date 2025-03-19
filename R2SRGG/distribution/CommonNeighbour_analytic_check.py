# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 1-11-2024
This .m is simulation for Maksim's mean field explanation.
We generate two nodes with fixed coordinates(0.495,0.5),(0.505,0.5), find all their common neighbours
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import json
import math
import sys

from scipy import integrate
from scipy.optimize import curve_fit

# from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair,dist_to_geodesic_perpendicular_R2
from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair, \
    dist_to_geodesic_perpendicular_R2
from R2SRGG.distribution.common_neighbour import common_neighbour_generator
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs


def compute_common_neighbour_deviation(G,Coorx,Coory,N):
    nodei = N - 2
    nodej = N - 1
    # Find the common neighbours
    common_neighbors = list(nx.common_neighbors(G, nodei, nodej))
    if common_neighbors:
        xSource = Coorx[nodei]
        ySource = Coory[nodei]
        xEnd = Coorx[nodej]
        yEnd = Coory[nodej]
        # length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd)) # for test
        # Compute deviation for the shortest path of each node pair
        deviations_for_a_nodepair = []
        for SPnode in common_neighbors:
            xMed = Coorx[SPnode]
            yMed = Coory[SPnode]
            # dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
            dist, _ = dist_to_geodesic_perpendicular_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
            deviations_for_a_nodepair.append(dist)
    else:
        deviations_for_a_nodepair = []
    return common_neighbors, deviations_for_a_nodepair


def neighbour_distance_with_ED_one_graph_clu(ED_index, beta_index):
    """
    From the results from neighbour_distance_beta(), we observed that the deviation of common neighbours grows with
    the increment of beta, which contradict the results according to the deviation of the shortest path
    The function are investigating why this contradiction appear.
    The main idea is:
    1. freeze the coordinates of the graph
    2. place i and j in (0.49),(0.5),(0.5),(0.5)
    3. for every beta, get the list of the common neighbours and see what will change
    4. do deviation for both "from a point to the geodesic" and "from a point to the line that geodesic belongs to(perpendicular distance)"
    :return:
    """
    N = 10000
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.2,2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4,5,6,7, 8,9,10, 16, 32, 64, 128]
    ED  = kvec[ED_index]
    Geodistance_index = 0
    distance_list = [[0.495, 0.5, 0.505, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5],
                     [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 2)
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    inputbeta_network = 2.2
    ExternalSimutime = 0
    network_index = 0
    beta = betavec[beta_index]
    print("ED:",ED)
    # load initial network

    filefolder_name = "/home/zqiu1/GSPP/SSRGGpy/R2/distribution/NetworkSRGG/"
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\"
    coorx = []
    coory = []
    FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}networktime{nt}.txt".format(
        Nn=N, EDn=5, betan=inputbeta_network, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime, nt=network_index)

    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            coorx.append(float(data[0]))
            coory.append(float(data[1]))
    common_neighbors_dic = {}
    deviations_for_a_nodepair_dic = {}
    connectedornot_dic = {}
    for simu_times in range(100):
        print(simu_times)
        G, coorx, coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B, coorx, coory)
        if nx.has_path(G,N-1,N-2):
            connectedornot_dic[simu_times] = 1
        else:
            connectedornot_dic[simu_times] = 0
        common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
        print("node",common_neighbors)
        print("dev", deviations_for_a_nodepair)
        common_neighbors_dic[simu_times] = common_neighbors
        deviations_for_a_nodepair_dic[simu_times] = deviations_for_a_nodepair


    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\perpendiculardistance\\foronegraph\\"
    common_neigthbour_name = filefolder_name2 + "common_neigthbour_list_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(common_neigthbour_name, 'w') as file:
        json.dump({str(k): v for k, v in common_neighbors_dic.items()}, file)

    deviations_name = filefolder_name2 + "common_neigthbour_deviationlist_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(deviations_name, 'w') as file:
        json.dump({str(k): v for k, v in deviations_for_a_nodepair_dic.items()}, file)

    connected_deviations_name = filefolder_name2 + "common_neigthbour_connection_list_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(connected_deviations_name, 'w') as file:
        json.dump({str(k): v for k, v in connectedornot_dic.items()}, file)


def neighbour_distance_with_ED_clu(ED_index, beta_index):
    """
    From the results from neighbour_distance_beta(), we observed that the deviation of common neighbours grows with
    the increment of beta, which contradict the results according to the deviation of the shortest path
    The function are investigating why this contradiction appear.
    The main idea is:
    1. freeze the coordinates of the graph
    2. place i and j in (0.49),(0.5),(0.5),(0.5)
    3. for every beta, get the list of the common neighbours and see what will change
    4. do deviation for both "from a point to the geodesic" and "from a point to the line that geodesic belongs to(perpendicular distance)"
    :return:
    """
    N = 10000
    kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    betavec = [2.2,2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4,5,6,7, 8,9,10, 16, 32, 64, 128]
    ED  = kvec[ED_index]
    Geodistance_index = 0
    distance_list = [[0.495, 0.5, 0.505, 0.5], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.3, 0.3], [0.25, 0.25, 0.5, 0.5],
                     [0.25, 0.25, 0.75, 0.75]]
    x_A = distance_list[Geodistance_index][0]
    y_A = distance_list[Geodistance_index][1]
    x_B = distance_list[Geodistance_index][2]
    y_B = distance_list[Geodistance_index][3]
    geodesic_distance_AB = round(x_B - x_A, 2)
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    ExternalSimutime = 0
    beta = betavec[beta_index]
    print("ED:",ED)
    # load initial network

    common_neighbors_dic = {}
    deviations_for_a_nodepair_dic = {}
    connectedornot_dic = {}
    for simu_times in range(100):
        print(simu_times)
        G, coorx, coory = R2SRGG_withgivennodepair(N, ED, beta, rg, x_A, y_A, x_B, y_B)
        if nx.has_path(G,N-1,N-2):
            connectedornot_dic[simu_times] = 1
        else:
            connectedornot_dic[simu_times] = 0
        common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
        print("node",common_neighbors)
        print("dev", deviations_for_a_nodepair)
        common_neighbors_dic[simu_times] = common_neighbors
        deviations_for_a_nodepair_dic[simu_times] = deviations_for_a_nodepair


    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\neighbour_distance\\perpendiculardistance\\"
    common_neigthbour_name = filefolder_name2 + "common_neigthbour_list_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(common_neigthbour_name, 'w') as file:
        json.dump({str(k): v for k, v in common_neighbors_dic.items()}, file)

    deviations_name = filefolder_name2 + "common_neigthbour_deviationlist_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(deviations_name, 'w') as file:
        json.dump({str(k): v for k, v in deviations_for_a_nodepair_dic.items()}, file)

    connected_deviations_name = filefolder_name2 + "common_neigthbour_connection_list_N{Nn}ED{EDn}beta{betan}xA{xA}yA{yA}xB{xB}yB{yB}Simu{simu}.json".format(
        Nn=N, EDn=ED, betan=beta, xA=x_A, yA=y_A, xB=x_B, yB=y_B, simu=ExternalSimutime)
    with open(connected_deviations_name, 'w') as file:
        json.dump({str(k): v for k, v in connectedornot_dic.items()}, file)


def check_probability_of_a_node_is_common_neighbour_node(N,avg,beta,delta):
    x_coords = np.random.uniform(-0.5, 0.5, 10000)
    y_coords = np.random.uniform(-0.5, 0.5, 10000)
    x_coords[9998] = -delta
    y_coords[9998] = 0
    x_coords[9999] = delta
    y_coords[9999] = 0
    rg = RandomGenerator(-12)
    # N = 10000
    # avg = 100
    # beta = 4
    G, coorx, coory = common_neighbour_generator(N, avg, beta, rg, x_coords, y_coords)
    common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
    p_simu = len(common_neighbors)/N
    # print(len(common_neighbors)/N, np.mean(deviations_for_a_nodepair))

    # delta = 0.005
    R = 2
    alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
    probability = compute_probability(alpha, beta, delta)
    # print(f"Pr[Ω=1] ≈ {probability:.6f}")
    return p_simu, probability


def conditional_probability(x, y, alpha, beta, delta):
    """
    probability that node k(x,y) is a common neighbour of node i and j, given (x,y)
    :param x:
    :param y:
    :param alpha:
    :param beta:
    :param delta:
    :return:
    """
    term1 = 1 + (alpha * np.sqrt((x + delta) ** 2 + y ** 2)) ** beta
    term2 = 1 + (alpha * np.sqrt((x - delta) ** 2 + y ** 2)) ** beta
    return 1 / (term1 * term2)

def compute_probability(alpha, beta, delta):
    """
    probability that node k is a common neighbour of node i and j
    :param x:
    :param y:
    :param alpha:
    :param beta:
    :param delta:
    :return:
    """
    result, _ = integrate.dblquad(
        conditional_probability,  # Function to integrate
        -0.5, 0.5,               # Limits for x
        lambda x: -0.5, lambda x: 0.5,  # Limits for y
        args=(alpha, beta, delta)  # Additional parameters
    )
    return result

def check_absolute_y():
    """
    int_{-1}^1 |y| f_Y(y) dy = 2
    when we remove the absolute value, the pdf of y f_Y(y) does not change
    :return:
    """
    random_number = np.random.uniform(0, 0.5,100000)
    print(np.mean(abs(random_number)))

def integrand(x, y,alpha,beta,delta):
    return conditional_probability(x, y, alpha, beta, delta) * y

def compute_Expected_abs_y(alpha,beta,delta):
    integral_result, _ = integrate.dblquad(integrand, 0, 1/2, lambda y: -1/2, lambda y: 1/2, args=(alpha,beta,delta))
    Pr_Omega = compute_probability(alpha, beta, delta)  # 请替换为 Pr[Ω=1] 的值
    try:
        final_result = (2 / Pr_Omega) * integral_result
    except:
        final_result = 0
    return final_result

def check_Expected_abs_y(N,avg,beta,delta,simutime):
    simu_res_vec =[]
    for i in range(simutime):
        x_coords = np.random.uniform(-0.5, 0.5, 10000)
        y_coords = np.random.uniform(-0.5, 0.5, 10000)
        x_coords[9998] = -delta
        y_coords[9998] = 0
        x_coords[9999] = delta
        y_coords[9999] = 0
        rg = RandomGenerator(-12)
        G, coorx, coory = common_neighbour_generator(N, avg, beta, rg, x_coords, y_coords)
        common_neighbors, deviations_for_a_nodepair = compute_common_neighbour_deviation(G, coorx, coory, N)
        # print(len(common_neighbors))
        # p_simu = len(common_neighbors) / N
        if deviations_for_a_nodepair:
            simu_res = np.mean(deviations_for_a_nodepair)
            simu_res_vec.append(simu_res)
    # print(len(common_neighbors)/N, np.mean(deviations_for_a_nodepair))

    simu_res_mean = (np.mean(simu_res_vec))
    # delta = 0.005
    R = 2
    alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
    ana_res = compute_Expected_abs_y(alpha,beta,delta)
    # print(ana_res)
    return ana_res,simu_res_mean

def power_law(x, a, k):
    return a * x ** k


def check_Expected_abs_y_alpha(N,avg,beta,delta):
    # delta = 0.005
    R = 2
    alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
    ana_res = compute_Expected_abs_y(alpha,beta,delta)
    # print(ana_res)
    return ana_res


def check_modelwithdistance():
    avg_vec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    beta_vec = [128]
    for delta in [0.005,0.01,0.05,0.1,0.15,0.25]:
        ana_vec = []
        simu_vec = []
        for avg in avg_vec:
            for beta in beta_vec:
                print("ED", avg, beta)
                ana_res,simu_res = check_Expected_abs_y(10000, avg, beta, delta,simutime=0)
                print("ana:",ana_res)
                print("sim:",simu_res)
                ana_vec.append(ana_res)
                simu_vec.append(simu_res)
        # print(ana_vec)
        # params, covariance = curve_fit(power_law, avg_vec[8:15], ana_vec[8:15])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(avg_vec[8:15], power_law(avg_vec[8:15], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
        #          color='red')

        plt.plot(avg_vec,ana_vec,label = f"{delta}")
        # plt.plot(avg_vec,simu_vec)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation of common neighbour model', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend()
    plt.show()

def check_model_withalpha():
    # # check \alpha:
    avg_vec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    beta_vec = [4]
    N = 10000
    R=2
    for delta in [0.005,0.25]:
        ana_vec = []

        alpha_vec =[]
        for avg in avg_vec:
            for beta in beta_vec:
                print("ED", avg, beta)
                ana_res = check_Expected_abs_y_alpha(N, avg, beta, delta)
                print("ana:",ana_res)

                ana_vec.append(ana_res)

                alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
                alpha = math.sqrt(alpha)
                alpha_vec.append(alpha)

        # print(ana_vec)
        # params, covariance = curve_fit(power_law, avg_vec[8:15], ana_vec[8:15])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(avg_vec[8:15], power_law(avg_vec[8:15], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
        #          color='red')

        plt.plot(alpha_vec,ana_vec,label = f"{delta}")
        # plt.plot(avg_vec,simu_vec)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation of common neighbour model', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend()
    plt.show()


def check_modelwithN():
    beta = 4
    delta_vec = [0.005,0.26]
    for delta in delta_vec:
        for N in [100]:
            log_spaced_points = np.logspace(np.log10(5), np.log10(N - 1), num=20)
            # Round the points to the nearest integer
            rounded_points_499 = np.round(log_spaced_points).astype(int)
            # Remove duplicates to ensure unique values
            avg_vec = sorted(set(rounded_points_499))
            min = avg_vec[-2]
            log_spaced_points = np.logspace(np.log10(min), np.log10(N - 1), num=50)
            # Round the points to the nearest integer
            rounded_points_499 = avg_vec+list(np.round(log_spaced_points).astype(int))
            # Remove duplicates to ensure unique values
            avg_vec= sorted(set(rounded_points_499))


            ana_vec = []
            simu_vec = []
            for avg in avg_vec:
                print("ED", avg, beta)
                ana_res,simu_res = check_Expected_abs_y(N, avg, beta, delta,simutime=0)
                print("ana:",ana_res)
                print("sim:",simu_res)
                ana_vec.append(ana_res)
                simu_vec.append(simu_res)
            # print(ana_vec)

            if delta == 0.26:
                params, covariance = curve_fit(power_law, avg_vec[-50:], ana_vec[-50:])
                # 获取拟合的参数
                a_fit, k_fit = params
                print(f"拟合结果: a = {a_fit}, k = {k_fit}")
                plt.plot(avg_vec[-50:], power_law(avg_vec[-50:], *params), linewidth=5, label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
                         color='red')
            else:
                params, covariance = curve_fit(power_law, avg_vec[-65:], ana_vec[-65:])
                # 获取拟合的参数
                a_fit, k_fit = params
                plt.plot(avg_vec[-65:], power_law(avg_vec[-65:], *params), linewidth=5,
                         label=f'fit curve: $y={a_fit:.6f}x^{{{k_fit:.4f}}}$',
                         color='yellow')

            plt.plot(avg_vec,ana_vec,label = f"{N},{delta_vec}")
            # plt.plot(avg_vec,simu_vec)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average deviation of common neighbour model', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend()
    plt.show()




if __name__ == '__main__':

    """
    check neighbour distance 
    """
    # neighbour_distance_with_beta_one_graph()
    # N = 10000
    # for k in range(2,10):
    #     print(math.sqrt(k/(math.pi*N)))
    """
    check neighbour distance for clu
    """
    # EDindex = sys.argv[1]
    # betaindex = sys.argv[2]
    # # betaindex = 0
    # neighbour_distance_with_ED_one_graph_clu(int(EDindex),int(betaindex))

    """
    check how analytic result changes with neighbour distance
    """
    # Simu_vec = []
    # Analaytic_result = []
    # avg_vec = [5, 10, 20, 100]
    # beta_vec = [2.2, 4, 8, 64, 128]
    # for avg in avg_vec:
    #     for beta in beta_vec:
    #         print(avg, beta)
    #         for i in range(1000):
    #             p_simu, probability = check_probability_of_a_node_is_common_neighbour_node(10000,avg,beta,0.01)
    #             Simu_vec.append(p_simu)
    #             Analaytic_result.append(probability)
    #         print("simu:",np.mean(Simu_vec))
    #         print("analytic:", np.mean(Analaytic_result))
    #
    # check_absolute_y()



    """
    check alpha
    """
    # check_model_withalpha()

    """
    check alpha
    """
    # check_model_withalpha()

    """"
    check model with N
    """

    check_modelwithN()
