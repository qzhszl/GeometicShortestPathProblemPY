# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 4-9-2024
"""
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from R2SRGG.R2SRGG import R2SRGG, loadSRGGandaddnode
from SphericalSoftRandomGeomtricGraph import RandomGenerator


def generate_and_plot_SRGG(N, avg, beta):
    # Figure 10 Appendix onset of GCC
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()
    # Coorx = []
    # Coory = []
    # FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #     Nn=N, EDn=4, betan=4)
    # with open(FileNetworkCoorName, "r") as file:
    #     for line in file:
    #         if line.startswith("#"):
    #             continue
    #         data = line.strip().split("\t")
    #         Coorx.append(float(data[0]))
    #         Coory.append(float(data[1]))
    G, xx, yy = R2SRGG(N, avg, beta, rg, SaveNetworkPath=None)

    # FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #                 Nn=N, EDn=avg, betan=beta)
    # nx.write_edgelist(G, FileNetworkName)
    # FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #     Nn=N, EDn=avg, betan=beta)
    # with open(FileNetworkCoorName, "w") as file:
    #     for data1, data2 in zip(xx, yy):
    #         file.write(f"{data1}\t{data2}\n")


    clustering_coefficient = nx.average_clustering(G)
    print("real cc:", clustering_coefficient)

    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)

    pos = {i: (xx[i], yy[i]) for i in range(N)}

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    max_len = 0
    diameter_path = []

    for u in lengths.keys():
        for v in lengths[u].keys():
            if u != v and lengths[u][v] > max_len:
                max_len = lengths[u][v]
                diameter_path = nx.shortest_path(G, source=u, target=v)

    # 构造 diameter 边列表
    diameter_edges = list(zip(diameter_path, diameter_path[1:]))


    # 绘制图
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # plt.text(0.2, 1.2, r'$\mathbb{E}[D] = 2, \mathbb{E}[D] = 5, \mathbb{E}[D] = 10$',
    #          transform=ax.transAxes,
    #          fontsize=22,
    #          bbox=dict(facecolor='white', alpha=0.5))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    nx.draw_networkx_edges(G, pos, edge_color='#C89FBF', width=3)

    nx.draw_networkx_edges(G, pos, edgelist=diameter_edges, edge_color='#62ABC7', width=5)
    # 绘制空心节点
    nx.draw_networkx_nodes(G, pos,
                           node_size=80,
                           node_color='none',  # 空心节点
                           edgecolors='#7A7DB1',  # 圆圈颜色
                           linewidths=5)

    plt.axis('equal')  # 保持横纵比例一致，图形不会变形
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    # picname = f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\SRGG_N{N}_ED{avg}_avg{real_avg}.pdf"
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    picname = f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\SRGG_N{N}_ED{avg}_avg{real_avg}.svg"

    # picname = f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\SRGG_legend.svg"

    plt.savefig(picname, format='svg', bbox_inches='tight', transparent = True)

    plt.show()



def generate_and_plot_SRGG_2(N, avg, beta):
    # Figure 2
    rg = RandomGenerator(-12)
    rseed = random.randint(0, 100)
    for i in range(rseed):
        rg.ran1()
    FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=avg, betan=beta)
    G = loadSRGGandaddnode(N, FileNetworkName)
    # load coordinates with noise
    Coorx = []
    Coory = []
    FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
        Nn=N, EDn=avg, betan=beta)
    with open(FileNetworkCoorName, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")
            Coorx.append(float(data[0]))
            Coory.append(float(data[1]))
    xx = Coorx
    yy = Coory


    # G, xx, yy = R2SRGG(N, avg, beta, rg, SaveNetworkPath=None)
    #
    # FileNetworkName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\network_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #                 Nn=N, EDn=avg, betan=beta)
    # nx.write_edgelist(G, FileNetworkName)
    # FileNetworkCoorName = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #     Nn=N, EDn=avg, betan=beta)
    # with open(FileNetworkCoorName, "w") as file:
    #     for data1, data2 in zip(xx, yy):
    #         file.write(f"{data1}\t{data2}\n")



    clustering_coefficient = nx.average_clustering(G)
    print("real cc:", clustering_coefficient)

    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("real ED:", real_avg)

    pos = {i: (xx[i], yy[i]) for i in range(N)}

    nodes = 69
    nodet = 73
    diameter_path = nx.shortest_path(G,nodes,nodet)

    # lengths = dict(nx.all_pairs_shortest_path_length(G))
    # max_len = 0
    # diameter_path = []
    #
    # for u in lengths.keys():
    #     for v in lengths[u].keys():
    #         if u != v and lengths[u][v] > max_len:
    #             max_len = lengths[u][v]
    #             diameter_path = nx.shortest_path(G, source=u, target=v)
    #
    # # 构造 diameter 边列表
    diameter_edges = list(zip(diameter_path, diameter_path[1:]))


    # 绘制图
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # plt.text(0.2, 1.2, r'$\mathbb{E}[D] = 2, \mathbb{E}[D] = 5, \mathbb{E}[D] = 10$',
    #          transform=ax.transAxes,
    #          fontsize=22,
    #          bbox=dict(facecolor='white', alpha=0.5))

    colors = ["#D08082", "#C89FBF", "#62ABC7", "#7A7DB1", '#6FB494']

    nx.draw_networkx_edges(G, pos, edge_color='#C89FBF', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=diameter_edges, edge_color='#6FB494', width=5)

    # node_colors = ['#62ABC7' if n in diameter_path else '#7A7DB1' for n in G.nodes]
    # node_sizes = [40 if n in diameter_path else 30 for n in G.nodes]

    # 绘制空心节点
    nx.draw_networkx_nodes(G, pos,
                           node_size=30,
                           node_color='#7A7DB1',  # 空心节点
                           edgecolors='#7A7DB1',  # 圆圈颜色
                           linewidths=6)

    nx.draw_networkx_nodes(G, pos, nodelist=diameter_path, node_size=50, node_color='#6FB494',edgecolors='#6FB494')
    # nx.draw_networkx_labels(G, pos)
    # text = r"$\mathbb{E}[D] = 3.5$"
    text = r"$N = 150\\$, $\beta = 4$,\n $\mathbb{E}[D] = 12$"
    text = r"$N = 150$" +"\n"+ r"$\beta = 4$" + "\n" + r"$\mathbb{E}[D] = 12$"
    ax.text(
        0.5, 1,  # 文本位置（轴坐标，0.5 表示图中央，1.05 表示轴上方）
        text,
        transform=ax.transAxes,  # 使用轴坐标
        fontsize=26,  # 字体大小
        ha='left',  # 水平居中对齐
        va='bottom'  # 垂直对齐方式
    )

    plt.axis('equal')  # 保持横纵比例一致，图形不会变形
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    # picname = f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\SRGG_N{N}_ED{avg}_avg{real_avg}.pdf"
    # plt.savefig(picname, format='pdf', bbox_inches='tight', dpi=600)
    picname = f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\SRGG_N{N}_ED{avg}_avg{real_avg}.svg"

    # picname = f"D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\example network\\SRGG_legend.svg"

    plt.savefig(picname, format='svg', bbox_inches='tight', transparent = True)

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
    # generate_and_plot_SRGG(200,10,4)
    generate_and_plot_SRGG_2(150, 12, 8)

    # generate_plot_model_graph()

