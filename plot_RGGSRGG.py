# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 4-9-2024
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from R2SRGG.R2SRGG import R2SRGG
from SphericalSoftRandomGeomtricGraph import RandomGenerator


def generate_and_plot_SRGG(N, avg, beta):
    rg = RandomGenerator(-12)
    G, xx, yy = R2SRGG(N, avg, beta, rg, Coorx=None, Coory=None, SaveNetworkPath=None)

    clustering_coefficient = nx.average_clustering(G)
    print("real cc:", clustering_coefficient)

    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)

    pos = {i: (xx[i], yy[i]) for i in range(N)}
    # 绘制图
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black',
            edge_color='gray')
    title_name = "N{Nn}ED{EDn}beta{betaN}"
    plt.title(title_name)
    plt.show()

def generate_plot_model_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2),(2,3),(3,4),(2,4)])  # using a list of edge tuples
    nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black',
            edge_color='gray')

    clustering_coefficient = nx.average_clustering(G)
    print("real cc:", clustering_coefficient)

    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generate_and_plot_SRGG(20,12,180)
    # generate_plot_model_graph()

