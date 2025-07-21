# -*- coding UTF-8 -*-

import numpy as np
import networkx as nx


precision_RGG_Name_side = "/home/zqiu1/data/PrecisionRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
        EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
np.savetxt(precision_RGG_Name_side, Precision_RGG_nodepair_side)

recall_RGG_Name_side = "/home/zqiu1/data/RecallRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
np.savetxt(recall_RGG_Name_side, Recall_RGG_nodepair_side)

precision_Geodis_Name_side = "/home/zqiu1/data/PrecisionGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
np.savetxt(precision_Geodis_Name_side, Precision_Geodis_nodepair_side)

recall_Geodis_Name_side = "/home/zqiu1/data/RecallGeodisED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
np.savetxt(recall_Geodis_Name_side, Recall_Geodis_nodepair_side)

precision_SRGG_Name_side = "/home/zqiu1/data/PrecisionSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
np.savetxt(precision_SRGG_Name_side, Precision_SRGG_nodepair_side)

recall_SRGG_Name_side = "/home/zqiu1/data/RecallSRGGED{EDn}Beta{betan}Noise{no}PYSimu{ST}.txt".format(
    EDn=ED, betan=beta, no=noise_amplitude, ST=ExternalSimutime)
np.savetxt(recall_SRGG_Name_side, Recall_SRGG_nodepair_side)


# G = nx.Graph()
# G.add_edges_from([
#     (1, 2), (2, 3), (3, 4), (4, 5), (6, 5),(6,7)
# ])
# G.add_node(7)



# # Define the two nodes
# nodei = 2
# nodej = 7
#
# for i in range(10):
#     # Get neighbors of i and j
#     neighbors_i = set(G.neighbors(nodei))
#     neighbors_j = set(G.neighbors(nodej))
#
#     # Union of both neighbor sets
#     combined_neighbors = neighbors_i.union(neighbors_j)
#     print(nx.shortest_path_length(G,nodei,nodej))
#     print(combined_neighbors)
#
#     combined_neighbors = set()
#     SP_list_set = set()
#     try:
#         distance = nx.shortest_path_length(G, nodei, nodej)
#
#
#     if distance > 1:
#         # Find all the neighbours of node i and node j
#         neighbors_i = set(G.neighbors(nodei))
#         neighbors_j = set(G.neighbors(nodej))
#
#         # Union of both neighbor sets
#         combined_neighbors = neighbors_i.union(neighbors_j)
#
#         # 预先计算所有节点到 nodei 和 nodej 的最短路径长度
#         lengths_from_nodei = nx.single_source_shortest_path_length(G, nodei)
#         lengths_from_nodej = nx.single_source_shortest_path_length(G, nodej)
#
#         for nodek in combined_neighbors:
#             d1 = lengths_from_nodei.get(nodek)
#             d2 = lengths_from_nodej.get(nodek)
#             if d1 is not None and d2 is not None and d1 + d2 == distance:
#                 SP_list_set.add(nodek)
#     print(SP_list_set)


