# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/11/17
This file is for the minimum of the shortest path
we first prove the degree is because of the relative deviation decrease for the shortest path(about k and s)
"""
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import networkx as nx
from R2SRGG.R2SRGG import loadSRGGandaddnode, distR2
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import find_k_connected_node_pairs


def check_meandistancebetweentwonodes():
    """
    This function simulated the average distance between arbitrary two nodes
    :return:
    """

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 10000)
    for i in range(rseed):
        rg.ran1()
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    N = 10000
    beta = 4
    ED = 9999
    # load a network
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    G = loadSRGGandaddnode(N, FileNetworkName)
    # load coordinates with noise
    Coorx = []
    Coory = []

    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\inputavgbeta\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))

    simu1 = []
    unique_pairs = find_k_connected_node_pairs(G, 1000)
    count = 0
    for node_pair in unique_pairs:
        count = count + 1
        nodei = node_pair[0]
        nodej = node_pair[1]
        xSource = Coorx[nodei]
        ySource = Coory[nodei]
        xEnd = Coorx[nodej]
        yEnd = Coory[nodej]
        simu1.append(distR2(xSource, ySource, xEnd, yEnd))

    print(np.mean(simu1))
    simu2 = []
    for simultime in range(1000):
        node1, node2 = random.sample(range(G.number_of_nodes()), 2)
        nodei = node1
        nodej = node2
        xSource = Coorx[nodei]
        ySource = Coory[nodei]
        xEnd = Coorx[nodej]
        yEnd = Coory[nodej]
        simu2.append(distR2(xSource, ySource, xEnd, yEnd))
    print("1:", np.mean(simu1))
    print("2:", np.mean(simu2))

def real_ave_geodesic_length_in_simu():
    kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    beta = 4
    exemptionlist = []
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        real_ave_degree_vec = []
        for beta in [beta]:
            for ED in kvec:
                ave_deviation_for_a_para_comb = np.array([])
                for ExternalSimutime in range(50):
                    try:
                        deviation_vec_name = filefolder_name + "length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb = np.hstack(
                            (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))
    print(exemptionlist)
    plt.xscale('log')
    plt.plot(kvec, ave_deviation_vec)
    plt.show()


def filter_data_from_hop_geo_dev(N, ED, beta):
    """
    :param N: network size of SRGG
    :param ED: expected degree of the SRGG
    :param beta: temperature parameter of the SRGG
    :return: a dict: key is hopcount, value is list for relative deviation = ave deviation of the shortest paths nodes for a node pair / geodesic of the node pair
    """
    exemptionlist = []
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"
    hopcount_for_a_para_comb = np.array([])
    geodistance_for_a_para_comb = np.array([])
    ave_deviation_for_a_para_comb = np.array([])

    ExternalSimutimeNum = 100
    for ExternalSimutime in range(ExternalSimutimeNum):
        try:
            # load data for hopcount for all node pairs
            hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
            hopcount_for_a_para_comb = np.hstack(
                (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))

            # load data for geo distance for all node pairs
            geodistance_vec_name = filefolder_name + "length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            geodistance_for_a_para_comb_10times = np.loadtxt(geodistance_vec_name)
            geodistance_for_a_para_comb = np.hstack(
                (geodistance_for_a_para_comb, geodistance_for_a_para_comb_10times))
            # load data for ave_deviation for all node pairs
            deviation_vec_name = filefolder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
            ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)

            ave_deviation_for_a_para_comb = np.hstack(
                (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
        except FileNotFoundError:
            exemptionlist.append((N, ED, beta, ExternalSimutime))

    hopcount_for_a_para_comb = hopcount_for_a_para_comb[hopcount_for_a_para_comb>1]
    counter = Counter(hopcount_for_a_para_comb)
    print(counter)

    relative_dev = ave_deviation_for_a_para_comb/geodistance_for_a_para_comb
    hop_relativedev_dict = {}
    for key in np.unique(hopcount_for_a_para_comb):
        hop_relativedev_dict[key] = relative_dev[hopcount_for_a_para_comb == key].tolist()
    return hop_relativedev_dict

def relative_deviation_vsdiffk():
    """
    plot the relative deviation of the shortest path
    :return:
    """
    # kvec = [6.0, 10, 16, 27, 44, 72, 118]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    kvec = [6.0, 10, 16, 27, 44, 72, 118]

    # kvec = [6.0, 10, 16, 27, 44,72,118]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    beta = 4
    exemptionlist = []
    fixed_hop = 5.0
    for N in [10000]:
        ave_deviation_vec = []
        std_deviation_vec = []
        ave_relative_deviation_vec = []
        std_relative_deviation_vec = []

        for beta in [beta]:
            for ED in kvec:
                print("ED:", ED)
                #------------------------------------------------- PLOT how the deviation change
                ave_deviation_for_a_para_comb = np.array([])
                for ExternalSimutime in range(100):
                    try:
                        deviation_vec_name = filefolder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
                            Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                        ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                        ave_deviation_for_a_para_comb = np.hstack(
                            (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
                    except FileNotFoundError:
                        exemptionlist.append((N, ED, beta, ExternalSimutime))

                ave_deviation_vec.append(np.mean(ave_deviation_for_a_para_comb))
                std_deviation_vec.append(np.std(ave_deviation_for_a_para_comb))

                #----------------------------------------------------
                hop_relativedev_dict = filter_data_from_hop_geo_dev(N, ED, beta)

                print(hop_relativedev_dict.keys())

                relative_dev_vec_foroneparacombine = hop_relativedev_dict[fixed_hop]
                print("DATA NUM:",len(relative_dev_vec_foroneparacombine))
                ave_relative_deviation_vec.append(np.mean(relative_dev_vec_foroneparacombine))
                std_relative_deviation_vec.append(np.std(relative_dev_vec_foroneparacombine))

    print(exemptionlist)
    # plot the figure of deviation
    plt.figure()
    plt.xscale('log')
    plt.plot(kvec, ave_deviation_vec)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    fixed_hop = int(fixed_hop)
    plt.errorbar(kvec, ave_relative_deviation_vec,std_relative_deviation_vec, label=f'hopcount: {fixed_hop}')
    plt.xscale('log')
    plt.xlabel('E[D]', fontsize=26)
    plt.ylabel('Average relative deviation of shortest path', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=20, loc="lower right")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "LocalMinimum_relative_ev_vs_avg_beta{beta}.pdf".format(beta=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def relative_deviation_vsdiffk_diffhop():
    """
    plot the relative deviation of the shortest path
    :return:
    """
    # colors = [[0, 0.4470, 0.7410],
    #           [0.8500, 0.3250, 0.0980],
    #           [0.9290, 0.6940, 0.1250],
    #           [0.4940, 0.1840, 0.5560],
    #           [0.4660, 0.6740, 0.1880]]

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # kvec = [6.0, 10, 16, 27, 44, 72, 118]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = [6.0, 10, 16, 27, 44, 72, 118]

    kvec = [6.0,8,10, 12,16, 20,27,34,44,56,72,92,118]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"

    beta = 4
    exemptionlist = []
    N = 10000
    fixed_hop_vec = [4.0, 8.0, 12.0, 16.0, 20.0]
    fig, ax = plt.subplots(figsize=(12, 8))
    colorindex = 0
    for fixed_hop in fixed_hop_vec:
        ave_deviation_vec = []
        std_deviation_vec = []
        ave_relative_deviation_vec = []
        std_relative_deviation_vec = []

        for beta in [beta]:
            for ED in kvec:
                print("ED:", ED)
                #----------------------------------------------------
                hop_relativedev_dict = filter_data_from_hop_geo_dev(N, ED, beta)

                print(hop_relativedev_dict.keys())
                try:
                    relative_dev_vec_foroneparacombine = hop_relativedev_dict[fixed_hop]
                    print("DATA NUM:",len(relative_dev_vec_foroneparacombine))
                    ave_relative_deviation_vec.append(np.mean(relative_dev_vec_foroneparacombine))
                    std_relative_deviation_vec.append(np.std(relative_dev_vec_foroneparacombine))
                except:
                    pass

        fixed_hop = int(fixed_hop)

        plt.errorbar(kvec[0:len(ave_relative_deviation_vec)-1], ave_relative_deviation_vec[0:len(ave_relative_deviation_vec)-1], std_relative_deviation_vec[0:len(ave_relative_deviation_vec)-1],
                     label=f'hopcount: {fixed_hop}', color=colors[colorindex], linestyle="--", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=16)
        colorindex = colorindex +1

    plt.xscale('log')
    plt.xlabel('E[D]', fontsize=32)
    plt.ylabel('Relative deviation', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc="best")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "LocalMinimum_relative_dev_vs_avg_beta{beta}.pdf".format(beta=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()



if __name__ == '__main__':
    # """
    # simulation of the average distance if we pick up an arbitrary node pair from an SRGG
    # """
    # check_meandistancebetweentwonodes()
    #
    # """
    #  simulation of the average distance in our simulation
    # """
    # real_ave_geodesic_length_in_simu()

    """
    check the relative deviation of the shortest path
    """
    # filter_data_from_hop_geo_dev(10000,5.0,4)
    relative_deviation_vsdiffk()
    # relative_deviation_vsdiffk_diffhop()

