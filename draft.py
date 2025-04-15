import numpy as np
import networkx as nx
import random
import math
import sys
import os
import shutil

from R2SRGG.R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs, hopcount_node




def distance_inSRGG_clu(network_size_index, average_degree_index, beta_index, ExternalSimutime):
    Nvec = [10, 20, 50, 100, 200, 500, 1000, 10000]
    # kvec = list(range(2, 16)) + [20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    # kvec = [2,2.5,3,3.5,4,4.5,5,5.5,6]
    kvec = [5.0, 5.6, 6.0, 10, 16, 27, 44, 72, 118, 193]
    # kvec = np.arange(2, 6.1, 0.2)
    # kvec = [round(a, 1) for a in kvec]
    # kvec = np.arange(6.5, 9.6, 0.5)
    # kvec = [round(a, 1) for a in kvec]
    # kvec2 = [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3727, 6105, 9999]
    # kvec = kvec + kvec2
    # kvec = [5,10,20]
    # kvec = np.arange(2.5, 5, 0.1)
    # kvec = [round(a, 1) for a in kvec]
    # kvec = [8,12,20,34,56,92]

    # kvec = [15,16]

    # kvec = [5,20]
    betavec = [2.2, 4, 8, 16, 32, 64, 128]
    # betavec = [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,3.8,3.9]
    # betavec = [2.2, 3.0, 4.2, 5.9, 8.3, 11.7, 16.5, 23.2, 32.7, 46.1, 64.9, 91.5, 128.9, 181.7, 256]


    random.seed(ExternalSimutime)
    N = Nvec[network_size_index]
    ED = kvec[average_degree_index]
    beta = betavec[beta_index]
    print("input para:", (N, ED, beta))

    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()

    # for large network, we only generate one network and randomly selected 1,000 node pair.
    # for small network, we generate 100 networks and selected all the node pair in the LCC
    if N > 100:
        distance_inlargeSRGG_clu(N, ED, beta,rg, ExternalSimutime)
    else:
        # Random select nodepair_num nodes in the largest connected component
        pass

def distance_inlargeSRGG_clu(N,ED,beta,rg,ExternalSimutime):
    """
    :param N:
    :param ED:
    :param beta:
    :param rg:
    :param ExternalSimutime:
    :return:
    for each node pair, we record the ave,max,min of distance from the shortest path to the geodesic,
    length of the geo distances.
    The generated network, the selected node pair and all the deviation of both shortest path and baseline nodes will be recorded.
    """
    if N> ED:
        deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
        # For each node pair:
        ave_deviation = []
        max_deviation = []
        min_deviation = []
        ave_baseline_deviation =[]
        length_geodesic = []
        hopcount_vec = []
        max_dev_node_hopcount = []
        corresponding_sp_max_dev_node_hopcount = []
        SPnodenum_vec =[]
        LCC_vec =[]
        second_vec = []

        source_folder = "/home/zqiu1/GSPP_SRGG_Dev/NetworkSRGG/"
        # ?????
        destination_folder = "/work/zqiu1/"
        network_template = "network_N{Nn}ED{EDn}Beta{betan}.txt"
        networkcoordinate_template  = "network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt"

        # load or generate a network
        try:
            FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            G = loadSRGGandaddnode(N, FileNetworkName)
            # load coordinates with noise
            Coorx = []
            Coory = []

            FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            with open(FileNetworkCoorName, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    data = line.strip().split("\t")
                    Coorx.append(float(data[0]))
                    Coory.append(float(data[1]))
        # except:
        #     os.makedirs(destination_folder, exist_ok=True)
        #     source_file = source_folder+ network_template.format(Nn=N, EDn=ED, betan=beta)
        #     destination_file = destination_folder+ network_template.format(Nn=N, EDn=ED, betan=beta)
        #     shutil.copy(source_file, destination_file)
        #     print(f"Copied: {source_file} -> {destination_file}")
        #     source_file = source_folder + networkcoordinate_template.format(Nn=N, EDn=ED, betan=beta)
        #     destination_file = destination_folder + networkcoordinate_template.format(Nn=N, EDn=ED, betan=beta)
        #     shutil.copy(source_file, destination_file)
        #     print(f"Copied: {source_file} -> {destination_file}")
        #
        #     FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #         Nn=N, EDn=ED, betan=beta)
        #     G = loadSRGGandaddnode(N, FileNetworkName)
        #     # load coordinates with noise
        #     Coorx = []
        #     Coory = []
        #
        #     FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
        #         Nn=N, EDn=ED, betan=beta)
        #     with open(FileNetworkCoorName, "r") as file:
        #         for line in file:
        #             if line.startswith("#"):
        #                 continue
        #             data = line.strip().split("\t")
        #             Coorx.append(float(data[0]))
        #             Coory.append(float(data[1]))

        except:
            G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
            # if ExternalSimutime == 0:
            #     FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            #         Nn=N, EDn=ED, betan=beta)
            #     nx.write_edgelist(G, FileNetworkName)
            #     FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
            #         Nn=N, EDn=ED, betan=beta)
            #     with open(FileNetworkCoorName, "w") as file:
            #         for data1, data2 in zip(Coorx, Coory):
            #             file.write(f"{data1}\t{data2}\n")

        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        print("real ED:", real_avg)

        # Randomly choose 100 connectede node pairs
        nodepair_num = 100
        unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
        # filename_selecetednodepair = "/work/zqiu1/selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")

        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(connected_components) > 1:
            largest_largest_component = connected_components[0]
            largest_largest_size = len(largest_largest_component)
            LCC_vec.append(largest_largest_size)
            # ?????????????????
            second_largest_component = connected_components[1]
            second_largest_size = len(second_largest_component)
            second_vec.append(second_largest_size)
        # if ExternalSimutime==0:
        #     filefolder_name = "/work/zqiu1/"
        #     LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
        #         Nn=N, EDn=ED, betan=beta)
        #     with open(LCCname, "w") as file:
        #         file.write("# LCC\tSECLCC\n")
        #         for name, age in zip(LCC_vec, second_vec):
        #             file.write(f"{name}\t{age}\n")

        for node_pair in unique_pairs:
            print("node_pair:",node_pair)
            nodei = node_pair[0]
            nodej = node_pair[1]
            # Find the shortest path nodes
            sp_test = nx.shortest_path(G, nodei, nodej)
            SPNodelist = all_shortest_path_node(G, nodei, nodej)
            hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))
            SPnodenum = len(SPNodelist)
            SPnodenum_vec.append(SPnodenum)
            if SPnodenum>0:
                xSource = Coorx[nodei]
                ySource = Coory[nodei]
                xEnd = Coorx[nodej]
                yEnd = Coory[nodej]
                length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
                # Compute deviation for the shortest path of each node pair
                deviations_for_a_nodepair = []
                for SPnode in SPNodelist:
                    xMed = Coorx[SPnode]
                    yMed = Coory[SPnode]
                    dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                    deviations_for_a_nodepair.append(dist)

                deviation_vec = deviation_vec+deviations_for_a_nodepair

                ave_deviation.append(np.mean(deviations_for_a_nodepair))
                max_deviation.append(max(deviations_for_a_nodepair))
                min_deviation.append(min(deviations_for_a_nodepair))


                # max hopcount
                max_value = max(deviations_for_a_nodepair)
                max_index = deviations_for_a_nodepair.index(max_value)
                maxhop_node_index = SPNodelist[max_index]
                max_dev_node_hopcount.append(hopcount_node(G, nodei, nodej, maxhop_node_index))
                a1 = nx.shortest_path_length(G, nodei, maxhop_node_index)
                a2 = nx.shortest_path_length(G, nodej, maxhop_node_index)

                corresponding_sp_max_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, nodej))

                # baseline: random selected
                # baseline_deviations_for_a_nodepair = []
                # # compute baseline's deviation
                # filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
                # base_line_node_index = random.sample(filtered_numbers,SPnodenum)
                #
                # for SPnode in base_line_node_index:
                #     xMed = Coorx[SPnode]
                #     yMed = Coory[SPnode]
                #     dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                #     baseline_deviations_for_a_nodepair.append(dist)
                # ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
                # baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair

        # deviation_vec_name = "/work/zqiu1/deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(deviation_vec_name, deviation_vec)
        # # baseline_deviation_vec_name = "/work/zqiu1/deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        # #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # # np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
        # # For each node pair:
        # ave_deviation_name = "/work/zqiu1/ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(ave_deviation_name, ave_deviation)
        # max_deviation_name = "/work/zqiu1/max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(max_deviation_name, max_deviation)
        # min_deviation_name = "/work/zqiu1/min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(min_deviation_name, min_deviation)
        # ave_baseline_deviation_name = "/work/zqiu1/ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        # length_geodesic_name = "/work/zqiu1/length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(length_geodesic_name, length_geodesic)
        # SPnodenum_vec_name = "/work/zqiu1/SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
        # hopcount_Name = "/work/zqiu1/hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #             Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(hopcount_Name, hopcount_vec)

        # max_dev_node_hopcount_name = "/work/zqiu1/max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")
        # max_dev_node_hopcount_name2 = "/work/zqiu1/sphopcountmax_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(max_dev_node_hopcount_name2, corresponding_sp_max_dev_node_hopcount, fmt="%i")



# distance_inSRGG_clu(7, 2, 2, 0)
filename = "network_N10000ED193Beta4.txt"
G =  loadSRGGandaddnode(10000,filename)
real_avg = 2*nx.number_of_edges(G)/nx