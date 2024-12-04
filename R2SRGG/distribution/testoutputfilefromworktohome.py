import itertools

import numpy as np
import networkx as nx
import random
import math
import sys
import os
import shutil

from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode
from SphericalSoftRandomGeomtricGraph import RandomGenerator
from main import all_shortest_path_node, find_k_connected_node_pairs, find_all_connected_node_pairs



def copy_data_to_slashwork(source_file, destination_file):
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file does not exist: {source_file}")

    # 检查目标目录是否存在
    destination_dir = os.path.dirname(destination_file)
    if not os.path.exists(destination_dir):
        print(f"Destination directory does not exist, creating: {destination_dir}")
        os.makedirs(destination_dir, exist_ok=True)

    # 检查是否有写入权限
    if not os.access(destination_dir, os.W_OK):
        raise PermissionError(f"Do not have write permissions for: {destination_dir}")

    # 复制文件
    shutil.copy(source_file, destination_file)
    print(f"File copied successfully from {source_file} to {destination_file}")

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
        SPnodenum_vec =[]
        LCC_vec =[]
        second_vec = []

        source_folder = "/home/zqiu1/GSPP_SRGG_Dev/NetworkSRGG/"
        # # 目标文件夹
        filenetwork_template = "network_N{Nn}ED{EDn}Beta{betan}.txt"
        filecoor_template =  "network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt"
        source_network = source_folder + filenetwork_template.format(Nn=N, EDn=ED, betan=beta)
        source_coordinate = source_folder + filecoor_template.format(Nn=N, EDn=ED, betan=beta)


        FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        FileNetworkCoorName = "/work/zqiu1/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
            Nn=N, EDn=ED, betan=beta)
        # load or generate a network
        try:
            G = loadSRGGandaddnode(N, FileNetworkName)
            # load coordinates with noise
            Coorx = []
            Coory = []
            with open(FileNetworkCoorName, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    data = line.strip().split("\t")  # 使用制表符分割
                    Coorx.append(float(data[0]))
                    Coory.append(float(data[1]))

        except:
            copy_data_to_slashwork(source_network, FileNetworkName)
            copy_data_to_slashwork(source_coordinate, FileNetworkCoorName)
            G = loadSRGGandaddnode(N, FileNetworkName)
            # load coordinates with noise
            Coorx = []
            Coory = []
            with open(FileNetworkCoorName, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    data = line.strip().split("\t")  # 使用制表符分割
                    Coorx.append(float(data[0]))
                    Coory.append(float(data[1]))

            # G, Coorx, Coory = R2SRGG(N, ED, beta, rg)


        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        print("real ED:", real_avg)

        # Randomly choose 100 connectede node pairs
        nodepair_num = 50
        unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
        filename_selecetednodepair = "/work/zqiu1/selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")

        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(connected_components) > 1:
            largest_largest_component = connected_components[0]
            largest_largest_size = len(largest_largest_component)
            LCC_vec.append(largest_largest_size)
            # 获取第二大连通分量的节点集合和大小
            second_largest_component = connected_components[1]
            second_largest_size = len(second_largest_component)
            second_vec.append(second_largest_size)
        if ExternalSimutime==0:
            filefolder_name = "/work/zqiu1/"
            LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
                Nn=N, EDn=ED, betan=beta)
            with open(LCCname, "w") as file:
                file.write("# LCC\tSECLCC\n")  # 使用制表符分隔列
                # 写入数据
                for name, age in zip(LCC_vec, second_vec):
                    file.write(f"{name}\t{age}\n")

        for node_pair in unique_pairs:
            print("node_pair:",node_pair)
            nodei = node_pair[0]
            nodej = node_pair[1]
            # Find the shortest path nodes
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

                baseline_deviations_for_a_nodepair = []
                # compute baseline's deviation
                filtered_numbers = [num for num in range(N) if num not in [nodei,nodej]]
                base_line_node_index = random.sample(filtered_numbers,SPnodenum)

                for SPnode in base_line_node_index:
                    xMed = Coorx[SPnode]
                    yMed = Coory[SPnode]
                    dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                    baseline_deviations_for_a_nodepair.append(dist)
                ave_baseline_deviation.append(np.mean(baseline_deviations_for_a_nodepair))
                baseline_deviation_vec = baseline_deviation_vec + baseline_deviations_for_a_nodepair

        deviation_vec_name = "/work/zqiu1/deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(deviation_vec_name, deviation_vec)
        baseline_deviation_vec_name = "/work/zqiu1/deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
        # For each node pair:
        ave_deviation_name = "/work/zqiu1/ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_deviation_name, ave_deviation)
        # max_deviation_name = "/work/zqiu1/max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(max_deviation_name, max_deviation)
        # min_deviation_name = "/work/zqiu1/min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
        #     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        # np.savetxt(min_deviation_name, min_deviation)
        ave_baseline_deviation_name = "/work/zqiu1/ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
        length_geodesic_name = "/work/zqiu1/length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(length_geodesic_name, length_geodesic)
        SPnodenum_vec_name = "/work/zqiu1/SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
            Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(SPnodenum_vec_name, SPnodenum_vec,fmt="%i")
        hopcount_Name = "/work/zqiu1/hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
                    Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
        np.savetxt(hopcount_Name, hopcount_vec)
        
if __name__ == '__main__':
    rg = RandomGenerator(-12)
    distance_inlargeSRGG_clu(10000,2.2,64,rg,0)