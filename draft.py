deviation_vec = []  # deviation of all shortest path nodes for all node pairs
baseline_deviation_vec = []  # deviation of all shortest path nodes for all node pairs
# For each node pair:
ave_deviation = []
max_deviation = []
min_deviation = []
ave_edge_length = []
ave_baseline_deviation = []
length_geodesic = []
length_edge_vec = []
hopcount_vec = []
max_dev_node_hopcount = []
corresponding_sp_max_dev_node_hopcount = []
SPnodenum_vec = []
LCC_vec = []
second_vec = []
delta_vec = []  # delta is the Euclidean geometric distance between two nodes i,k, where i,k is the neighbours of j

folder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\deviaitonvsSPgeometriclength\\"

try:
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    G = loadSRGGandaddnode(N, FileNetworkName)
    # load coordinates with noise
    Coorx = []
    Coory = []

    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\max_min_ave_ran_deviation\\largenetwork\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))
except:
    G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
    FileNetworkName = folder_name + "network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    nx.write_edgelist(G, FileNetworkName)
    FileNetworkCoorName = folder_name + "network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=ED, betan=beta)
    with open(FileNetworkCoorName, "w") as file:
        for data1, data2 in zip(Coorx, Coory):
            file.write(f"{data1}\t{data2}\n")

# G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
# #     # if ExternalSimutime == 0:
# FileNetworkName = folder_name+"network_N{Nn}ED{EDn}Beta{betan}.txt".format(
#     Nn=N, EDn=ED, betan=beta)
# nx.write_edgelist(G, FileNetworkName)
# FileNetworkCoorName = folder_name+"network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
#     Nn=N, EDn=ED, betan=beta)
# with open(FileNetworkCoorName, "w") as file:
#     for data1, data2 in zip(Coorx, Coory):
#         file.write(f"{data1}\t{data2}\n")

real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
print("real ED:", real_avg)

real_avg_name = folder_name + "real_avg_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(real_avg_name, real_avg)

# Randomly choose 100 connectede node pairs
nodepair_num = 10000
unique_pairs = find_k_connected_node_pairs(G, nodepair_num)
# filename_selecetednodepair = folder_name+"selected_node_pair_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
# np.savetxt(filename_selecetednodepair, unique_pairs, fmt="%i")

# LCC and the second LCC
# connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
# if len(connected_components) > 1:
#     largest_largest_component = connected_components[0]
#     largest_largest_size = len(largest_largest_component)
#     LCC_vec.append(largest_largest_size)
#     # ?????????????????
#     second_largest_component = connected_components[1]
#     second_largest_size = len(second_largest_component)
#     second_vec.append(second_largest_size)
# if ExternalSimutime==0:
#     filefolder_name = folder_name+""
#     LCCname = filefolder_name + "LCC_2LCC_N{Nn}ED{EDn}beta{betan}.txt".format(
#         Nn=N, EDn=ED, betan=beta)
#     with open(LCCname, "w") as file:
#         file.write("# LCC\tSECLCC\n")
#         for name, age in zip(LCC_vec, second_vec):
#             file.write(f"{name}\t{age}\n")
count = 0
for node_pair in unique_pairs:
    count = count + 1
    print(f"{count}node_pair:{node_pair}")

    nodei = node_pair[0]
    nodej = node_pair[1]
    # Find the shortest path nodes
    try:
        SPNodelist = nx.shortest_path(G, nodei, nodej)
        SPnodenum = len(SPNodelist) - 2
        SPnodenum_vec.append(SPnodenum)
        if SPnodenum > 0:
            hopcount_vec.append(nx.shortest_path_length(G, nodei, nodej))

            # compute the length of the edges
            length_edge_for_anodepair = []
            shortest_path_edges = list(zip(SPNodelist[:-1], SPNodelist[1:]))
            for (nodes, nodet) in shortest_path_edges:
                d_E = compute_edge_Euclidean_length(nodes, nodet, Coorx, Coory)
                length_edge_for_anodepair.append(d_E)
            length_edge_vec = length_edge_vec + length_edge_for_anodepair
            ave_edge_length.append(np.mean(length_edge_for_anodepair))

            # compute the deviation
            xSource = Coorx[nodei]
            ySource = Coory[nodei]
            xEnd = Coorx[nodej]
            yEnd = Coory[nodej]
            length_geodesic.append(distR2(xSource, ySource, xEnd, yEnd))
            # Compute deviation for the shortest path of each node pair
            deviations_for_a_nodepair = []
            for SPnode in SPNodelist[1:len(SPNodelist) - 1]:
                xMed = Coorx[SPnode]
                yMed = Coory[SPnode]
                dist, _ = dist_to_geodesic_R2(xMed, yMed, xSource, ySource, xEnd, yEnd)
                deviations_for_a_nodepair.append(dist)

            deviation_vec = deviation_vec + deviations_for_a_nodepair

            ave_deviation.append(np.mean(deviations_for_a_nodepair))
            # max_deviation.append(max(deviations_for_a_nodepair))
            # min_deviation.append(min(deviations_for_a_nodepair))

            # Compute delta for the shortest path of each node pair:
            delta_for_a_nodepair = []
            for i in range(len(SPNodelist) - 2):  # 计算相隔节点的距离
                node1 = SPNodelist[i]
                node2 = SPNodelist[i + 2]

                delta = distR2(Coorx[node1], Coory[node1], Coorx[node2], Coory[node2])
                delta_for_a_nodepair.append(delta)

            delta_vec = delta_vec + delta_for_a_nodepair

            # max hopcount
            # max_value = max(deviations_for_a_nodepair)
            # max_index = deviations_for_a_nodepair.index(max_value)
            # maxhop_node_index = SPNodelist[max_index]
            # max_dev_node_hopcount.append(hopcount_node(G, nodei, nodej, maxhop_node_index))
            # corresponding_sp_max_dev_node_hopcount.append(nx.shortest_path_length(G, nodei, nodej))

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
    except:
        pass
deviation_vec_name = folder_name + "deviation_shortest_path_nodes_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(deviation_vec_name, deviation_vec)
# baseline_deviation_vec_name = folder_name+"deviation_baseline_nodes_num_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
# np.savetxt(baseline_deviation_vec_name, baseline_deviation_vec)
# For each node pair:
ave_deviation_name = folder_name + "ave_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(ave_deviation_name, ave_deviation)
# max_deviation_name = folder_name+"max_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
# np.savetxt(max_deviation_name, max_deviation)
# min_deviation_name = folder_name+"min_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
# np.savetxt(min_deviation_name, min_deviation)
# ave_baseline_deviation_name = folder_name+"ave_baseline_deviation_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#     Nn = N, EDn=ED, betan=beta, ST=ExternalSimutime)
# np.savetxt(ave_baseline_deviation_name, ave_baseline_deviation)
length_geodesic_name = folder_name + "length_geodesic_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(length_geodesic_name, length_geodesic)
SPnodenum_vec_name = folder_name + "SPnodenum_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(SPnodenum_vec_name, SPnodenum_vec, fmt="%i")
hopcount_Name = folder_name + "hopcount_sp_N{Nn}_ED{EDn}Beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(hopcount_Name, hopcount_vec, fmt="%i")
delta_Name = folder_name + "delta_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(delta_Name, delta_vec)

edgelength_name = folder_name + "edgelength_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(edgelength_name, length_edge_vec)

aveedgelength_name = folder_name + "ave_edgelength_N{Nn}ED{EDn}beta{betan}Simu{ST}.txt".format(
    Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
np.savetxt(aveedgelength_name, ave_edge_length)

# max_dev_node_hopcount_name = folder_name+"max_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
# np.savetxt(max_dev_node_hopcount_name, max_dev_node_hopcount, fmt="%i")
# max_dev_node_hopcount_name2 = folder_name+"sphopcountmax_dev_node_hopcount_N{Nn}ED{EDn}Beta{betan}Simu{ST}.txt".format(
#     Nn=N, EDn=ED, betan=beta, ST=ExternalSimutime)
# np.savetxt(max_dev_node_hopcount_name2, corresponding_sp_max_dev_node_hopcount, fmt="%i")