# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 14-10-2024
"""
# import numpy as np
import networkx as nx
import numpy as np
import pandas as pd
# import random
# import math
import sys

# from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair
from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode, R2SRGG_withgivennodepair
from SphericalSoftRandomGeomtricGraph import RandomGenerator


def generate_proper_EDCC_network(N, input_ED, beta_index):
    # for 10000 node
    betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    beta = betavec[beta_index]
    cc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    C_G = cc[beta_index]

    kvec = list(range(6, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    ED_bound = input_ED*0.05

    min_ED = 1
    max_ED = N-1
    count = 0
    rg = RandomGenerator(-12)
    G, Coorx, Coory = R2SRGG(N, input_ED, beta, rg)
    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    ED= input_ED
    while abs(input_ED-real_avg)>ED_bound and count < 20:
        count = count + 1
        if input_ED-real_avg>0:
            min_ED = ED
            ED = min_ED+0.5*(max_ED-min_ED)
        else:
            max_ED = ED
            ED = min_ED + 0.5 * (max_ED - min_ED)
        G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)

    print("input para:", (N, input_ED, beta))
    print("real ED:", real_avg)
    print("clu:", nx.average_clustering(G))
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    print("LCC", len(largest_component))
    if count < 20:
        FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\network_N{Nn}ED{EDn}CC{betan}.txt".format(
            Nn=N, EDn=input_ED, betan=C_G)
        nx.write_edgelist(G, FileNetworkName)

        FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\network_coordinates_N{Nn}ED{EDn}CC{betan}.txt".format(
            Nn=N, EDn=input_ED, betan=C_G)
        with open(FileNetworkCoorName, "w") as file:
            for data1, data2 in zip(Coorx, Coory):
                file.write(f"{data1}\t{data2}\n")
        FileNetworkrealED = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\EDnetwork_N{Nn}inputED{EDn}CC{betan}.txt".format(
            Nn=N, EDn=input_ED, betan=C_G)
        np.savetxt(FileNetworkrealED,[ED])
    return ED


def load_proper_network_paras():
    kvec = list(range(6, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    cc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    N = 10000
    EDdic = {}
    beta_dic = {}
    betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    smallED_matrix = [[2.6, 2.6, 2.6, 2.6, 2.6, 0, 0],
                      [4.3, 4, 4, 4, 4, 0],
                      [5.6, 5.4, 5.2, 5.2, 5.2, 5.2, 5.2],
                      [7.5, 6.7, 6.5, 6.5, 6.5, 6.5]]
    smallbeta_matrix = [[3.1, 4.5, 6, 300, 0, 0],
                        [2.7, 3.5, 4.7, 7.6, 300, 0],
                        [2.7, 3.4, 4.3, 5.7, 11, 300],
                        [2.55, 3.2, 4, 5.5, 8.5, 300]]
    count = 0
    for C_G in cc:
        ED_vec=[smallED_matrix[0][count],smallED_matrix[1][count],smallED_matrix[2][count],smallED_matrix[3][count]]
        for input_ED in kvec:
            FileNetworkrealED = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\inputparameter\\EDnetwork_N{Nn}inputED{EDn}CC{betan}.txt".format(
                Nn=N, EDn=input_ED, betan=C_G)
            ED=np.loadtxt(FileNetworkrealED)
            ED = ED.tolist()
            ED_vec = ED_vec+[ED]
        # print(ED_vec)
        EDdic[C_G] = ED_vec
        count = count + 1
    print(EDdic)

    kvec2 = list(range(2, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    df = pd.DataFrame.from_dict(EDdic, orient='index', columns=kvec2)
    print(df)

    # 保存为 Excel 文件
    df.to_csv("D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\inputparameter\\InputED.csv", index=True)
    count = 0
    for C_G in cc:
        beta_vec=[smallbeta_matrix[0][count],smallbeta_matrix[1][count],smallbeta_matrix[2][count],smallbeta_matrix[3][count]]
        beta_vec = beta_vec+[betavec[count]*1 for i in range(len(kvec))]
        beta_dic[C_G] = beta_vec
        count = count + 1
    print(beta_dic)

    df2 = pd.DataFrame.from_dict(beta_dic, orient='index', columns=kvec2)
    print(df2)

    # 保存为 Excel 文件
    df2.to_csv(
        "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\inputparameter\\Inputbeta.csv",
        index=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # """
    # # generate_proper_network(N, ED)
    # """
    # C_G_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # betavec = [2.55, 3.2, 3.99, 5.15, 7.99, 300]
    # kvec = list(range(6, 20)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # count = 0
    #
    # ED = sys.argv[1]
    # betaindex = sys.argv[2]
    # ED_input = kvec[int(ED)]
    # real_input_ED = generate_proper_EDCC_network(10000, ED_input, int(betaindex))
    # print("real input ed:",real_input_ED)
    #
    """
    load_proper_network_paras, save the results in two 
    """
    # load_proper_network_paras()

    df = pd.read_csv(
        "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\cleanwithEDCC\\inputparameter\\InputED.csv",index_col=0, header=0)
    print(df.iloc[:, 4])

