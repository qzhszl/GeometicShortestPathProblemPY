# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/03/31
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
import pandas as pd



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


def filter_data_from_hop_geo_dev(N, ED, beta, datatype="onesp"):
    """
    :param N: network size of SRGG
    :param ED: expected degree of the SRGG
    :param beta: temperature parameter of the SRGG
    :return: a dict: key is hopcount, value is list for relative deviation = ave deviation of the shortest paths nodes for a node pair / geodesic of the node pair
    """
    exemptionlist = []
    if datatype == "allsp":
        filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # FOR all sp nodes
    else:
        filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # FOR ONE sp

    hopcount_for_a_para_comb = np.array([])
    geodistance_for_a_para_comb = np.array([])
    ave_deviation_for_a_para_comb = np.array([])

    ExternalSimutimeNum = 5
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

    # hop_relativedev_dict = {}
    # for key in np.unique(hopcount_for_a_para_comb):
    #     hop_relativedev_dict[key] = relative_dev[hopcount_for_a_para_comb == key].tolist()
    # return hop_relativedev_dict
    hop_dev_dict = {}
    hop_relativedev_dict = {}
    for key in np.unique(hopcount_for_a_para_comb):
        hop_dev_dict[key] = ave_deviation_for_a_para_comb[hopcount_for_a_para_comb == key].tolist()
        hop_relativedev_dict[key] = relative_dev[hopcount_for_a_para_comb == key].tolist()
    return hop_dev_dict, hop_relativedev_dict


def relative_deviation_vsdiffk():
    """
    plot the relative deviation of the shortest path with different average degree with given hopcount
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

    ave_dev_dict = {}
    std_dev_dict = {}

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']
    # kvec = [6.0, 10, 16, 27, 44, 72, 118]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = [6.0, 10, 16, 27, 44, 72, 118]

    kvec = [6.0,8,10, 12,16, 20,27,34,44,56,72,92,118]   # for all shortest path
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # for all shortest path

    kvec = [2,4,6,11,12,14,16,18,20,21,23,24,27,28,30,32,33,34,35,39,42,44,47,50,56,61,68,71,73,74,79,82,85,91,95,107,120,193,316,518,848,1389]  # for ONE SP
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP

    beta = 128
    exemptionlist = []
    N = 10000
    fixed_hop_vec = [3.0,4.0,5.0,6.0, 7.0, 8.0,9.0,10.0,11.0, 12.0, 16.0,20.0]
    fixed_hop_vec = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 16.0, 20.0,25.0,30.0,40.0]
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
                _,hop_relativedev_dict = filter_data_from_hop_geo_dev(N, ED, beta)

                print(hop_relativedev_dict.keys())
                try:
                    relative_dev_vec_foroneparacombine = hop_relativedev_dict[fixed_hop]
                    print("DATA NUM:",len(relative_dev_vec_foroneparacombine))
                    ave_relative_deviation_vec.append(np.mean(relative_dev_vec_foroneparacombine))
                    std_relative_deviation_vec.append(np.std(relative_dev_vec_foroneparacombine))
                except:
                    ave_relative_deviation_vec.append(np.nan)
                    std_relative_deviation_vec.append(np.nan)

        fixed_hop = int(fixed_hop)
        ave_dev_dict[fixed_hop] = ave_relative_deviation_vec
        std_dev_dict[fixed_hop] = std_relative_deviation_vec


        plt.errorbar(kvec[0:len(ave_relative_deviation_vec)-1], ave_relative_deviation_vec[0:len(ave_relative_deviation_vec)-1], std_relative_deviation_vec[0:len(ave_relative_deviation_vec)-1],
                     label=f'hopcount: {fixed_hop}', linestyle="--", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=16)


        colorindex += +1
    ave_df = pd.DataFrame(ave_dev_dict, index=kvec)
    std_df = pd.DataFrame(std_dev_dict, index=kvec)
    ave_df.index.name = None
    std_df.index.name = None
    ave_df.index = ave_df.index.astype(int)
    std_df.index = std_df.index.astype(int)
    print(ave_df)

    ave_df = ave_df.dropna(how='all')
    std_df = std_df.dropna(how = "all")

    ave_df.to_csv(filefolder_name + f"RelativeDeviation_Average_beta{beta}.csv",index = True,encoding='utf-8')
    std_df.to_csv(filefolder_name + f"RelativeDeviation_Std_beta{beta}.csv",index = True,encoding='utf-8')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=32)
    plt.ylabel('Relative deviation', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc="best",ncol=2)
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "LocalMinimum_relative_dev_vs_avg_beta{beta}2.pdf".format(beta=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def power_law(x, a, k):
    return a * x ** k


def linear_func(x, a, b):
    return a * x + b


def Deviation_vsdiffk_diffhop():
    """
    plot the deviation of the shortest path <d> vs E[D] Given different hopcount,
    i.e. when the hopcount is fixed, how the <d> changes with E[D]
    :return:
    """
    ave_dev_dict = {}
    std_dev_dict = {}

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494',"#A2C7A4","#9DB0C2","#E3B6A4"]
    # kvec = [6.0, 10, 16, 27, 44, 72, 118]
    # kvec = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = [6.0, 10, 16, 27, 44, 72, 118]

    # kvec = [6.0, 8, 10, 12, 16, 20, 27, 34, 44, 56, 72, 92, 118, 193]  # for all shortest path
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"  # for all shortest path

    kvec = [11, 12, 14, 16, 18, 21, 24, 27, 30, 33, 37, 42, 50, 56, 61, 68,
            73, 79, 85, 91, 107, 120,193]  # for ONE SP
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for ONE SP

    beta = 4
    exemptionlist = []
    N = 10000
    fixed_hop_vec = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0] # for beta = 4
    fixed_hop_vec = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0,14.0, 16.0]  # for beta = 4
    # fixed_hop_vec = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0]
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
                # ----------------------------------------------------
                hop_relativedev_dict, _ = filter_data_from_hop_geo_dev(N, ED, beta,"onesp")

                print(hop_relativedev_dict.keys())
                try:
                    relative_dev_vec_foroneparacombine = hop_relativedev_dict[fixed_hop]
                    print("DATA NUM:", len(relative_dev_vec_foroneparacombine))
                    if len(relative_dev_vec_foroneparacombine)>500:
                        ave_relative_deviation_vec.append(np.mean(relative_dev_vec_foroneparacombine))
                        std_relative_deviation_vec.append(np.std(relative_dev_vec_foroneparacombine))
                    else:
                        ave_relative_deviation_vec.append(np.nan)
                        std_relative_deviation_vec.append(np.nan)
                except:
                    ave_relative_deviation_vec.append(np.nan)
                    std_relative_deviation_vec.append(np.nan)

        fixed_hop = int(fixed_hop)
        ave_dev_dict[fixed_hop] = ave_relative_deviation_vec
        std_dev_dict[fixed_hop] = std_relative_deviation_vec

        plt.errorbar(kvec[0:len(ave_relative_deviation_vec) - 1],
                     ave_relative_deviation_vec[0:len(ave_relative_deviation_vec) - 1],
                     std_relative_deviation_vec[0:len(ave_relative_deviation_vec) - 1],
                     label=fr'$h$: {fixed_hop}', linestyle="--", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=16, color = colors[colorindex])
        colorindex += +1
    ave_df = pd.DataFrame(ave_dev_dict, index=kvec)
    std_df = pd.DataFrame(std_dev_dict, index=kvec)
    ave_df.index.name = None
    std_df.index.name = None
    ave_df.index = ave_df.index.astype(int)
    std_df.index = std_df.index.astype(int)
    print(ave_df)

    ave_df = ave_df.dropna(how='all')
    std_df = std_df.dropna(how="all")

    ave_df.to_csv(filefolder_name + f"Deviation_Average_beta{beta}.csv",index = True,encoding='utf-8')
    std_df.to_csv(filefolder_name + f"Deviation_Std_beta{beta}.csv",index = True,encoding='utf-8')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Expected degree, $E[D]$', fontsize=28)
    plt.ylabel(r'Average deviation, $\langle d \rangle$', fontsize=28)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30, loc="best", ncol=2)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = filefolder_name + "Dev_vs_avg_givenhop_beta{beta}.pdf".format(beta=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()



def Deviation_vs_diffk_diffhop_curvefit():
    beta = 128
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # for all the sp

    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for only one sp

    ave_df = pd.read_csv(filefolder_name + f"Deviation_Average_beta{beta}.csv", index_col=0)
    std_df=  pd.read_csv(filefolder_name + f"Deviation_Std_beta{beta}.csv", index_col=0)
    print(ave_df)
    fig, ax = plt.subplots(figsize=(12, 8))
    Avec =[]
    tau_vec =[]
    count = 0

    for column in ave_df.columns:
        plt.errorbar(ave_df.index,
                     ave_df[column],
                     std_df[column],
                     label=f'hopcount: {column}', linestyle="--", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=16)
        y = ave_df[column].dropna()
        x = ave_df.index[0:len(y)]

        a = 5
        b = -5
        # params, covariance = curve_fit(power_law, x[a:b], y[a:b])
        # # 获取拟合的参数
        # a_fit, k_fit = params

        Avec = [0.0029,0.0031,0.0033,0.0036,0.0038,0.004,0.0042,0.0044,0.0046,0.0048,0.0055,0.0062,0.007,0.0078,0.0095]
        tau_vec = [0.3 for i in range(len(Avec))]
        a_fit = Avec[count]
        k_fit = tau_vec[count]
        # a_fit = 0.0095
        # k_fit = 0.3

        params = np.array([a_fit, k_fit])

        print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        plt.plot(x[a:b], power_law(x[a:b], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.2f}}}$',
                         color='red')

        # params, covariance = curve_fit(power_law, x[a:], y[a:])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(x[a:], power_law(x[a:], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
        #          color='green')

        # if int(column)<7:
        #     params, covariance = curve_fit(power_law,x[a:b], y[a:b])
        #     # 获取拟合的参数
        #     a_fit, k_fit = params
        #     print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # else:
        #     params, covariance = curve_fit(power_law, x, y)
        #     # 获取拟合的参数
        #     a_fit, k_fit = params
        #     print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        count = count+1


    # plt.ylim([0.001,0.05])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E[D]', fontsize=32)
    plt.ylabel('Deviation', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.legend(fontsize=26, loc="best", ncol=2)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = filefolder_name + "dev_vs_avg_curve_fit{beta}hop{hop}.pdf".format(beta=beta,hop = column)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    avg = ave_df.columns
    avg = [int(i) for i in avg]
    # 第一个y轴（左侧）
    color1 = 'tab:blue'
    ax1.set_xlabel('Hopcount,h', fontsize=20)
    ax1.set_ylabel(r'$\tau$', color=color1, fontsize=20)
    ax1.plot(avg, np.abs(tau_vec), 'o-', color=color1, label=r'$\tau$')
    ax1.tick_params(axis='y', labelcolor=color1)

    # 第二个y轴（右侧）
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('A', color=color2, fontsize=20)
    ax2.plot(avg, Avec, 's--', color=color2, label='A')
    ax2.tick_params(axis='y', labelcolor=color2)

    a = 4
    params, covariance = curve_fit(power_law, avg[a:], Avec[a:])
    # 获取拟合的参数
    a_fit, k_fit = params
    print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    ax2.plot(avg[a:], power_law(avg[a:], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
             color='green')

    # 设置对数坐标轴
    ax2.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # 添加辅助text信息
    plt.text(0.6, 0.2, r'$f(h) = A(h)\cdot <k>^{\tau}$',
             transform=ax1.transAxes,
             fontsize=20,
             bbox=dict(facecolor='white', alpha=0.5))

    # 添加legend (需要合并两个轴的legend)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=18)

    # 标题和网格
    # plt.title('Dual Y-axis Log-Log Plot')
    # ax1.grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()





def relative_deviation_vs_diffk_diffhop_curvefit():
    beta = 4
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # for all the sp

    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for only one sp

    ave_df = pd.read_csv(filefolder_name + f"RelativeDeviation_Average_beta{beta}.csv", index_col=0)
    std_df=  pd.read_csv(filefolder_name + f"RelativeDeviation_Std_beta{beta}.csv", index_col=0)
    print(ave_df)
    fig, ax = plt.subplots(figsize=(12, 8))
    Avec =[]
    tau_vec =[]
    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494', "#A2C7A4", "#9DB0C2", "#E3B6A4"]

    count = 0
    # for column in ave_df.columns:
    for column in ["3", "4", "6", "8", "10", "12", "16","20"]:
        y = ave_df[column]
        error = std_df[column]
        x = ave_df.index
        if column not in ["3", "4", "6","20"]:
            y = y.dropna()[:-2]
            error = error.dropna()[:-2]
            x = x[:len(y)]
        plt.errorbar(x,
                     error,
                     error,
                     label=f'hopcount: {column}', linestyle="--", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=16,color = colors[count])
        y = ave_df[column].dropna()
        x = ave_df.index[0:len(y)]
        a = 3
        b = -3
        # params, covariance = curve_fit(power_law, x[a:b], y[a:b])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # Avec = [0.38,0.29,0.25,0.215,0.19,0.175,0.165,0.154,0.145,0.136,0.116,0.105,0.093,0.085,0.076]  # beta = 128
        # Avec = [0.44,0.36,0.315,0.29,0.28,0.27,0.26,0.25,0.24,0.23,0.000001,0.000001]  # beta = 4
        # tau_vec = [-0.3 for i in range(len(Avec))]
        # a_fit = Avec[count]
        # k_fit = tau_vec[count]
        # params = np.array([a_fit,k_fit])
        # if int(column) <10:
        #     plt.plot(x[a:b], power_law(x[a:b], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
        #                      color='red')
        # Avec.append(a_fit)
        # tau_vec.append(k_fit)
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # count = count+1

        # tau_vec = []  # beta = 4 one sp
        # Avec = []

        # params, covariance = curve_fit(power_law, x[a:], y[a:])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(x[a:], power_law(x[a:], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
        #          color='green')

        # if int(column)<7:
        #     params, covariance = curve_fit(power_law,x[a:b], y[a:b])
        #     # 获取拟合的参数
        #     a_fit, k_fit = params
        #     print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # else:
        #     params, covariance = curve_fit(power_law, x, y)
        #     # 获取拟合的参数
        #     a_fit, k_fit = params
        #     print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        count = count+1
    plt.ylim([0.001,0.5])
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(axis='both', which="both", length=6, width=1)
    plt.xlabel(r'Expected degree, $E[D]$', fontsize=28)
    plt.ylabel('Relative deviation', fontsize=28)
    plt.xticks([10,100],fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30, loc="best", ncol=2)
    picname = filefolder_name + "Relative_Dev_vs_avg_givenhop_beta{beta}.pdf".format(beta=beta)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    print(Avec)
    print(tau_vec)

    # fig, ax1 = plt.subplots(figsize=(8, 6))
    # avg = ave_df.columns
    # avg = [int(i) for i in avg]
    # # 第一个y轴（左侧）
    # color1 = 'tab:blue'
    # ax1.set_xlabel('Hopcount,h', fontsize=20)
    # ax1.set_ylabel(r'$\tau$', color=color1, fontsize=20)
    # ax1.plot(avg, np.abs(tau_vec), 'o-', color=color1, label=r'$\tau$')
    # ax1.tick_params(axis='y', labelcolor=color1)
    #
    # # 第二个y轴（右侧）
    # ax2 = ax1.twinx()
    # color2 = 'tab:red'
    # ax2.set_ylabel('A', color=color2, fontsize=20)
    # ax2.plot(avg, Avec, 's--', color=color2, label='A')
    # ax2.tick_params(axis='y', labelcolor=color2)
    #
    # # 设置对数坐标轴
    # ax2.set_xscale('log')
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    #
    # # 添加辅助text信息
    # plt.text(0.35, 0.7, r'$f(h) = A(h)\cdot <k>^{-\tau}$',
    #          transform=ax1.transAxes,
    #          fontsize=20,
    #          bbox=dict(facecolor='white', alpha=0.5))
    #
    # # 添加legend (需要合并两个轴的legend)
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=20)
    #
    # # 标题和网格
    # # plt.title('Dual Y-axis Log-Log Plot')
    # # ax1.grid(True, which='both', ls='--', alpha=0.5)
    #
    # plt.tight_layout()
    # plt.show()




def plot_relative_deviation_vs_diffk_diffhop_curvefit():
    beta = 4
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"
    ave_df = pd.read_csv(filefolder_name + f"RelativeDeviation_Average_beta{beta}.csv", index_col=0)
    avg = ave_df.columns
    avg = [int(i) for i in avg]
    # tau_vec = [-0.1986, -0.2149, -0.2013, -0.2169, -0.2404, -0.2018, -0.2334, -0.2368, -0.2464, -0.34, -0.359] # beta = 4 all
    # Avec = [0.3023, 0.2615, 0.2252, 0.2190, 0.2287, 0.1958, 0.2093, 0.2052, 0.2090, 0.25, 0.2487]
    # tau_vec = [-0.20, -0.22, -0.22, -0.23, -0.25, -0.22, -0.25, -0.25, -0.18, -0.19, -0.65]   # beta = 4 one sp
    # Avec = [0.32, 0.28, 0.25, 0.23, 0.24, 0.21, 0.23, 0.22, 0.18, 0.19, 0.55]

    # tau_vec = [-0.20, -0.22, -0.22, -0.23, -0.25, -0.22, -0.25, -0.25, -0.18, -0.19, -0.65]  # beta = 4 one sp
    # Avec = [0.32, 0.28, 0.25, 0.23, 0.24, 0.21, 0.23, 0.22, 0.18, 0.19, 0.55]
    # Avec = [0.38, 0.29, 0.25, 0.215, 0.19, 0.175, 0.165, 0.154, 0.145, 0.136, 0.116, 0.105, 0.093, 0.085, 0.076] # beta = 128
    Avec = [0.44, 0.36, 0.315, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21] # beta = 4
    tau_vec = [-0.3 for i in range(len(Avec))]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 第一个y轴（左侧）
    color1 = 'tab:blue'
    ax1.set_xlabel('Hopcount,h',fontsize=20)
    ax1.set_ylabel(r'$\tau$', color=color1,fontsize=20)
    ax1.plot(avg[:-5], np.abs(tau_vec[:-5]), 'o-', color=color1, label=r'$\tau$')
    ax1.tick_params(axis='y', labelcolor=color1)

    a = 0
    params, covariance = curve_fit(power_law, avg[a:-5], Avec[a:-5])
    # 获取拟合的参数
    a_fit, k_fit = params
    print(f"拟合结果: a = {a_fit}, k = {k_fit}")


    # 第二个y轴（右侧）
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('A', color=color2,fontsize=20)
    ax2.plot(avg[a:-5], power_law(avg[a:-5], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
             color='green')
    ax2.plot(avg[:-5], Avec[:-5], 's--', color=color2, label='A')
    ax2.tick_params(axis='y', labelcolor=color2)


    # 设置对数坐标轴
    ax2.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # 添加辅助text信息
    plt.text(0.5, 0.6, r'$f(h) = A(h)\cdot <k>^{-\tau}$',
             transform=ax1.transAxes,
             fontsize=20,
             bbox=dict(facecolor='white', alpha=0.5))

    # 添加legend (需要合并两个轴的legend)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',fontsize=20)

    # 标题和网格
    # plt.title('Dual Y-axis Log-Log Plot')
    # ax1.grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def relative_deviation_vs_diffk_diffhop_linearcurvefit():
    beta = 128
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\" # for all the sp

    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\OneSP\\"  # for only one sp

    ave_df = pd.read_csv(filefolder_name + f"RelativeDeviation_Average_beta{beta}.csv", index_col=0)
    std_df=  pd.read_csv(filefolder_name + f"RelativeDeviation_Std_beta{beta}.csv", index_col=0)
    print(ave_df)
    fig, ax = plt.subplots(figsize=(12, 8))
    Avec =[]
    tau_vec =[]

    ave_df = ave_df[:-7]
    std_df = std_df[:-7]
    count = 0
    for column in ave_df.columns:
        plt.plot(np.log(ave_df.index),
                     np.log(ave_df[column]),
                     label=f'hopcount: {column}', linestyle="--", linewidth=3, marker='o', markersize=16)
        y = ave_df[column].dropna()
        x = ave_df.index[0:len(y)]

        y = np.log(y)
        x = np.log(x)

        a = 4
        b = -5
        params, _ = curve_fit(linear_func, x[a:b], y[a:b])
        # 获取拟合的参数
        a_fit, k_fit = params


        # Avec = [0.38,0.29,0.25,0.215,0.19,0.175,0.165,0.154,0.145,0.136,0.116,0.105,0.093,0.085,0.076]
        # tau_vec = [-0.3 for i in range(len(Avec))]
        # a_fit = Avec[count]
        # k_fit = tau_vec[count]
        # params = np.array([a_fit,k_fit])
        plt.plot(x[a:b], linear_func(x[a:b], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
                         color='red')
        Avec.append(a_fit)
        tau_vec.append(k_fit)
        print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        count = count+1

        # tau_vec = []  # beta = 4 one sp
        # Avec = []

        # params, covariance = curve_fit(power_law, x[a:], y[a:])
        # # 获取拟合的参数
        # a_fit, k_fit = params
        # print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # plt.plot(x[a:], power_law(x[a:], *params), linewidth=5, label=f'fit curve: $y={a_fit:.4f}x^{{{k_fit:.4f}}}$',
        #          color='green')

        # if int(column)<7:
        #     params, covariance = curve_fit(power_law,x[a:b], y[a:b])
        #     # 获取拟合的参数
        #     a_fit, k_fit = params
        #     print(f"拟合结果: a = {a_fit}, k = {k_fit}")
        # else:
        #     params, covariance = curve_fit(power_law, x, y)
        #     # 获取拟合的参数
        #     a_fit, k_fit = params
        #     print(f"拟合结果: a = {a_fit}, k = {k_fit}")
    # plt.ylim([0.001,0.5])
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('E[D]', fontsize=32)
    plt.ylabel('Relative deviation', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.legend(fontsize=26, loc="best", ncol=2)
    plt.tick_params(axis='both', which="both", length=6, width=1)
    picname = filefolder_name + "Relative_dev_vs_avg_curve_fit{beta}hop{hop}.pdf".format(beta=beta,hop = column)
    plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    print(Avec)
    print(tau_vec)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    avg = ave_df.columns
    avg = [int(i) for i in avg]
    # 第一个y轴（左侧）
    color1 = 'tab:blue'
    ax1.set_xlabel('Hopcount,h', fontsize=20)
    ax1.set_ylabel(r'$\tau$', color=color1, fontsize=20)
    ax1.plot(avg, np.abs(Avec), 'o-', color=color1, label=r'$\tau$')
    ax1.tick_params(axis='y', labelcolor=color1)

    # 第二个y轴（右侧）
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('A', color=color2, fontsize=20)
    tau_vec = [np.exp(tau) for tau in tau_vec]
    ax2.plot(avg, tau_vec, 's--', color=color2, label='A')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 设置对数坐标轴
    ax2.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # 添加辅助text信息
    plt.text(0.35, 0.7, r'$f(h) = A(h)\cdot <k>^{-\tau}$',
             transform=ax1.transAxes,
             fontsize=20,
             bbox=dict(facecolor='white', alpha=0.5))

    # 添加legend (需要合并两个轴的legend)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=20)

    # 标题和网格
    # plt.title('Dual Y-axis Log-Log Plot')
    # ax1.grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()




def filter_data_from_hop_geo_dev_givendistance(N, ED, beta,geodesic_distance_AB):
    """
    :param N: network size of SRGG
    :param ED: expected degree of the SRGG
    :param beta: temperature parameter of the SRGG
    :return: a dict: key is hopcount, value is list for relative deviation = ave deviation of the shortest paths nodes for a node pair / geodesic of the node pair
    """
    exemptionlist = []
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\1000realization\\"
    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\1000realization\\tail\\"


    hopcount_for_a_para_comb = np.array([])
    geodistance_for_a_para_comb = np.array([])
    ave_deviation_for_a_para_comb = np.array([])

    ExternalSimutimeNum = 50
    for ExternalSimutime in range(ExternalSimutimeNum):
        try:
            # load data for hopcount for all node pairs
            hopcount_vec_name = filefolder_name2 + "Givendistancehopcount_sp_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)

            hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
            hopcount_for_a_para_comb = np.hstack(
                (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))

            # load data for geo distance for all node pairs
            geodistance_vec_name = filefolder_name2 + "Givendistancelength_geodesic_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)

            geodistance_for_a_para_comb_10times = np.loadtxt(geodistance_vec_name)
            geodistance_for_a_para_comb = np.hstack(
                (geodistance_for_a_para_comb, geodistance_for_a_para_comb_10times))
            # load data for ave_deviation for all node pairs
            deviation_vec_name = filefolder_name2 + "Givendistanceave_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
            ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)

            ave_deviation_for_a_para_comb = np.hstack(
                (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
        except FileNotFoundError:
            # load data for hopcount for all node pairs
            hopcount_vec_name = filefolder_name + "Givendistancehopcount_sp_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)

            hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
            hopcount_for_a_para_comb = np.hstack(
                (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))

            # load data for geo distance for all node pairs
            geodistance_vec_name = filefolder_name + "Givendistancelength_geodesic_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)

            geodistance_for_a_para_comb_10times = np.loadtxt(geodistance_vec_name)
            geodistance_for_a_para_comb = np.hstack(
                (geodistance_for_a_para_comb, geodistance_for_a_para_comb_10times))
            # load data for ave_deviation for all node pairs
            deviation_vec_name = filefolder_name + "Givendistanceave_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}Geodistance{Geodistance}.txt".format(
                Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime, Geodistance=geodesic_distance_AB)
            ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)

            ave_deviation_for_a_para_comb = np.hstack(
                (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))
        except:
            exemptionlist.append((N, ED, beta, ExternalSimutime))

    hopcount_for_a_para_comb = hopcount_for_a_para_comb[hopcount_for_a_para_comb>1]
    counter = Counter(hopcount_for_a_para_comb)
    print(counter)

    relative_dev = ave_deviation_for_a_para_comb/geodistance_for_a_para_comb
    hop_relativedev_dict = {}
    for key in np.unique(hopcount_for_a_para_comb):
        hop_relativedev_dict[key] = relative_dev[hopcount_for_a_para_comb == key].tolist()
    return hop_relativedev_dict


def relative_deviation_vsdiffk_diffhop_givendistance():
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

    kvec = [6.0,8,10,12,16,20,27,34,44,56,72,92,118]
    kvec = [5,6,7,8,9, 10, 16, 27, 44, 72, 118]

    kvec = [6, 8, 10, 16, 27, 44, 72, 118]
    # kvec = [10, 16, 27, 44, 72, 118]
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\1000realization\\"
    filefolder_name2 = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\GivenGeodistance\\1000realization\\tail\\"

    beta = 4
    Geodistance = 0.5
    exemptionlist = []
    N = 10000
    fixed_hop_vec = [4.0, 8.0, 12.0, 16.0]
    # fixed_hop_vec = [3.0]
    fig, ax = plt.subplots(figsize=(12, 8))
    colorindex = 0
    for fixed_hop in fixed_hop_vec:
        ave_deviation_vec = []
        std_deviation_vec = []
        ave_relative_deviation_vec = []
        std_relative_deviation_vec = []
        x = []
        for beta in [beta]:
            for ED in kvec:
                print("ED:", ED)
                #----------------------------------------------------
                hop_relativedev_dict = filter_data_from_hop_geo_dev_givendistance(N, ED, beta,Geodistance)

                print(hop_relativedev_dict.keys())
                try:
                    relative_dev_vec_foroneparacombine = hop_relativedev_dict[fixed_hop]
                    print("DATA NUM:",len(relative_dev_vec_foroneparacombine))
                    x.append(ED)
                    ave_relative_deviation_vec.append(np.mean(relative_dev_vec_foroneparacombine))
                    std_relative_deviation_vec.append(np.std(relative_dev_vec_foroneparacombine))
                except:
                    pass

        fixed_hop = int(fixed_hop)

        plt.errorbar(x[0:len(ave_relative_deviation_vec)-1], ave_relative_deviation_vec[0:len(ave_relative_deviation_vec)-1], std_relative_deviation_vec[0:len(ave_relative_deviation_vec)-1],
                     label=f'hopcount: {fixed_hop}', color=colors[colorindex], linestyle="--", linewidth=3, elinewidth=1,
                     capsize=5, marker='o', markersize=16)
        colorindex = colorindex +1

    # plt.text(2, 5, r"$d_{ij} = 0.5$", fontsize=26)
    text = r"$N = 10^4$,$\beta = 4$, $d_{ij} = 0.5$"
    plt.text(
        0.3, 0.85,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=26,  # 字体大小
        ha='center',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )


    plt.xscale('log')
    plt.xlabel('E[D]', fontsize=32)
    plt.ylabel('Relative deviation', fontsize=32)
    yticks = np.arange(0.03, 0.19, 0.03)
    plt.xticks(fontsize=26)
    plt.yticks(yticks,fontsize=26)
    plt.legend(fontsize=26, loc="best")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "LocalMinimum_relative_dev_vs_avg_beta{beta}_givendistance.pdf".format(beta=beta)
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()



def plot_dev_vs_hop(ED):
    """
    Not relative dev
    :param ED:
    :return:
    """
    # N = 10000
    # beta = 4
    # betavec = [2.1, 4, 8, 16, 32, 64, 128]
    exemptionlist = []
    filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\inpuavg_beta\\"
    hopcount_for_a_para_comb = np.array([])
    ave_deviation_for_a_para_comb = np.array([])

    for N in [10000]:
        for beta in [4]:
            for ExternalSimutime in range(100):
                try:
                    deviation_vec_name = filefolder_name + "ave_deviation_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    ave_deviation_for_a_para_comb_10times = np.loadtxt(deviation_vec_name)
                    ave_deviation_for_a_para_comb = np.hstack(
                        (ave_deviation_for_a_para_comb, ave_deviation_for_a_para_comb_10times))

                    hopcount_vec_name = filefolder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                        Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
                    hopcount_for_a_para_comb_10times = np.loadtxt(hopcount_vec_name)
                    hopcount_for_a_para_comb = np.hstack(
                        (hopcount_for_a_para_comb, hopcount_for_a_para_comb_10times))

                except FileNotFoundError:
                    exemptionlist.append((N, ED, beta, ExternalSimutime))

    hopcount_for_a_para_comb = hopcount_for_a_para_comb[hopcount_for_a_para_comb > 1]
    counter = Counter(hopcount_for_a_para_comb)
    print(counter)
    hop_relativedev_dict = {}
    for key in np.unique(hopcount_for_a_para_comb):
        hop_relativedev_dict[key] = ave_deviation_for_a_para_comb[hopcount_for_a_para_comb == key].tolist()

    x = []
    y = []
    error_v = []
    for key, vlist in hop_relativedev_dict.items():
        x.append(key)
        y.append(np.mean(vlist))
        error_v.append(np.std(vlist))

    plt.errorbar(x[:-2], y[:-2], yerr=error_v[:-2], linestyle="--", linewidth=3, elinewidth=1, capsize=5, marker='o',
                 markersize=16)
    # plt.xscale('log')
    # plt.yscale('log')
    xticks = [5,10,15,20]
    plt.xlabel('h', fontsize=26)
    plt.ylabel('Average deviation of shortest path', fontsize=20)
    plt.xticks(xticks, fontsize=26)
    plt.yticks(fontsize=26)
    plt.title(fr'N={N},ED={ED},$\beta$={beta}', fontsize=26)
    # plt.legend(fontsize=20, loc="lower right")
    plt.tick_params(axis='both', which="both", length=6, width=1)

    picname = filefolder_name + "dev_vs_h_N{N}avg{ed}_beta{beta}.pdf".format(N = N,ed = ED,beta=beta)
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
    # relative_deviation_vsdiffk()
    # relative_deviation_vsdiffk_diffhop()
    # relative_deviation_vs_diffk_diffhop_linearcurvefit()

    """
    curve fit
    """
    relative_deviation_vs_diffk_diffhop_curvefit()
    # plot_relative_deviation_vs_diffk_diffhop_curvefit()

    """
    draw the picture for given distance 
    """

    # relative_deviation_vsdiffk_diffhop_givendistance()

    """
    For a fixed <k>, say, <k> = 10, can you plot deviation as a function of hopcount
    """
    # plot_dev_vs_hop(10)


    """
    DEV VS diff ED given diff h  and dev vs h given diff ED(see plot_dev_vs_hop)
    """
    # Deviation_vsdiffk_diffhop()
    # Deviation_vs_diffk_diffhop_curvefit()

