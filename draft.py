# -*- coding UTF-8 -*-

import numpy as np
import networkx as nx

G = nx.Graph()
G.add_edges_from([
    (1, 2), (2, 3), (3, 4), (4, 5), (2, 5),(2,6)
])
G.add_node(7)
# Define the two nodes
nodei = 2
nodej = 7

for i in range(10):
    # Get neighbors of i and j
    neighbors_i = set(G.neighbors(nodei))
    neighbors_j = set(G.neighbors(nodej))

    # Union of both neighbor sets
    combined_neighbors = neighbors_i.union(neighbors_j)
    print(nx.shortest_path_length(G,nodei,nodej))
    print(combined_neighbors)

    combined_neighbors = set()
    SP_list_set = set()
    try:
        distance = nx.shortest_path_length(G, nodei, nodej)


    if distance > 1:
        # Find all the neighbours of node i and node j
        neighbors_i = set(G.neighbors(nodei))
        neighbors_j = set(G.neighbors(nodej))

        # Union of both neighbor sets
        combined_neighbors = neighbors_i.union(neighbors_j)

        # 预先计算所有节点到 nodei 和 nodej 的最短路径长度
        lengths_from_nodei = nx.single_source_shortest_path_length(G, nodei)
        lengths_from_nodej = nx.single_source_shortest_path_length(G, nodej)

        for nodek in combined_neighbors:
            d1 = lengths_from_nodei.get(nodek)
            d2 = lengths_from_nodej.get(nodek)
            if d1 is not None and d2 is not None and d1 + d2 == distance:
                SP_list_set.add(nodek)
    print(SP_list_set)


