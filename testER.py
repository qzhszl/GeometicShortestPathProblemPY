# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2025/12/1
"""
import math
import random

import networkx as nx
import matplotlib.pyplot as plt
from SphericalSoftRandomGeomtricGraph import RandomGenerator


def generate_geo_ER(N,prob,rg,Coorx=None, Coory=None):
    R = 2.0  # manually tuned value
    s = []
    t = []
    linkweight = []

    # Assign coordinates
    if Coorx is not None and Coory is not None:
        xx = Coorx
        yy = Coory
    else:
        xx = []
        yy = []
        for i in range(N):
            xx.append(rg.ran1())
            yy.append(rg.ran1())

    # make connections
    for i in range(N):
        for j in range(i + 1, N):
            if rg.ran1() < prob:
                s.append(i)
                t.append(j)
                dist = math.sqrt((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2)
                assert dist > 0
                linkweight.append(dist)


    # Create graph and remove self-loops
    G = nx.Graph()
    # G.add_edges_from(zip(s, t,linkweight))
    for nodei, nodej, dist in zip(s, t, linkweight):
        G.add_edge(nodei, nodej, weight=dist)

    max_edge_weight_per_node = {}

    for node in G.nodes():
        weights = [d["weight"] for _, _, d in G.edges(node, data=True)]
        if weights:  # 避免孤立节点报错
            max_edge_weight_per_node[node] = max(weights)
        else:
            max_edge_weight_per_node[node] = 0

    # 2. 求所有节点最大权重的平均值
    average_max_weight = sum(max_edge_weight_per_node.values()) / nx.number_of_nodes(G)

    if G.number_of_nodes() < N:
        ExpectedNodeList = [i for i in range(0, N)]
        Nodelist = list(G.nodes)
        difference = [item for item in ExpectedNodeList if item not in Nodelist]
        G.add_nodes_from(difference)

    return G, linkweight, average_max_weight, xx, yy


if __name__ == '__main__':
    rg = RandomGenerator(-12)
    for i in range(random.randint(1, 1000)):
        rg.ran1()
    N = 100
    prob = 0.3
    G, linkweight, average_max_weight, xx, yy = generate_geo_ER(N, prob, rg, Coorx=None, Coory=None)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)
    plt.show()

