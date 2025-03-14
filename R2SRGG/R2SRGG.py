# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 31-5-2024
Soft random geometric graph on 2-d Euclidean space
"""
import math
import random
import time

import networkx as nx
from numpy.random import random_integers

from SphericalSoftRandomGeomtricGraph import RandomGenerator
import numpy as np
# import matplotlib.pyplot as plt


# def point_to_line_distance_from_two_points(px, py, ax, ay, bx, by):
#     # 计算直线方程 Ax + By + C = 0 的系数
#     A = by - ay
#     B = ax - bx
#     C = (bx - ax) * ay - (by - ay) * ax
#
#     # 使用点到直线的距离公式
#     numerator = abs(A * px + B * py + C)
#     denominator = math.sqrt(A ** 2 + B ** 2)
#     distance = numerator / denominator
#
#     return distance
#
# def point_to_segment_distance(px, py, ax, ay, bx, by):
#     # 向量 AB 和 AP
#     AB = np.array([bx - ax, by - ay])
#     AP = np.array([px - ax, py - ay])
#
#     # 计算 AB 的平方长度
#     AB_squared = np.dot(AB, AB)
#
#     if AB_squared == 0:
#         # A 和 B 是同一个点
#         nearest_point = np.array([ax, ay])
#         distance = np.linalg.norm(AP)
#     else:
#         # 计算投影的比例 t
#         t = np.dot(AP, AB) / AB_squared
#
#         if t < 0:
#             # 投影在 A 点之前，最近点是 A
#             nearest_point = np.array([ax, ay])
#         elif t > 1:
#             # 投影在 B 点之后，最近点是 B
#             nearest_point = np.array([bx, by])
#         else:
#             # 投影在 A 和 B 之间，最近点是投影点
#             nearest_point = np.array([ax, ay]) + t * AB
#
#         # 返回点到最近点的距离
#         distance = np.linalg.norm(np.array([px, py]) - nearest_point)
#
#     return distance, nearest_point

def distR2(x1,y1,x2,y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


def dist_to_geodesic_R2(px, py, ax, ay, bx, by):
    # the distance between AB and an extra point P
    # return the distance and the nearest point between A and B
    # 向量 AB 和 AP
    AB = np.array([bx - ax, by - ay])
    AP = np.array([px - ax, py - ay])
    # 计算 AB 的平方长度
    AB_squared = np.dot(AB, AB)
    if abs(AB_squared) < 0.0000001:
        # A 和 B 是同一个点
        return np.linalg.norm(AP)
    # 计算投影的比例 t
    t = np.dot(AP, AB) / AB_squared

    if t < 0:
        # 投影在 A 点之前，最近点是 A
        nearest_point = np.array([ax, ay])
    elif t >= 1:
        # 投影在 B 点之后，最近点是 B
        nearest_point = np.array([bx, by])
    else:
        # 投影在 A 和 B 之间，最近点是投影点
        nearest_point = np.array([ax, ay]) + t * AB

    # 返回点到最近点的距离
    return np.linalg.norm(np.array([px, py]) - nearest_point), nearest_point


def dist_to_geodesic_perpendicular_R2(px, py, ax, ay, bx, by):
    # the distance between AB and an extra point P
    # return the distance and the nearest point between A and B
    # 向量 AB 和 AP
    AB = np.array([bx - ax, by - ay])
    AP = np.array([px - ax, py - ay])
    # 计算 AB 的平方长度
    AB_squared = np.dot(AB, AB)
    if abs(AB_squared) < 0.0000001:
        # A 和 B 是同一个点
        return np.linalg.norm(AP)
    # 计算投影的比例 t
    t = np.dot(AP, AB) / AB_squared
    nearest_point = np.array([ax, ay]) + t * AB

    # 返回点到最近点的距离
    return np.linalg.norm(np.array([px, py]) - nearest_point), nearest_point

def point_to_line_distance_from_points(x, y, xb, yb, xc, yc):
    # 计算直线BC的系数
    A = yc - yb
    B = xb - xc
    C = xc * yb - xb * yc

    # 计算点A到直线BC的距离
    numerator = abs(A * x + B * y + C)
    denominator = math.sqrt(A ** 2 + B ** 2)
    return numerator / denominator


def R2SRGG(N, avg, beta, rg, Coorx=None, Coory=None, SaveNetworkPath=None):
    """
    Program generates Soft Random Geometric Graph on a 2d unit square
    Connection probability function is (1 + (a*d)^{b})^{-1}
    :param N: number of nodes
    :param avg: <k>
    :param beta: beta (controlling clustering)
    :param rg:  random seed
    :param Coorx: coordinates of x
    :param Coory: coordinates of y
    :param SaveNetworkPath:
    :return:
    """
    # calculate parameters
    assert beta > 2
    assert avg > 0
    assert N > 1

    R = 2.0  # manually tuned value
    alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
    s = []
    t = []

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
            dist = math.sqrt((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2)
            assert dist > 0

            try:
                prob = 1 / (1 + math.exp(beta * math.log(alpha * dist)))
            except:
                prob = 0
            if rg.ran1() < prob:
                s.append(i)
                t.append(j)


    if SaveNetworkPath is not None:
        with open(SaveNetworkPath, "w") as file:
            for nodei, nodej in zip(s, t):
                file.write(f"{nodei}\t{nodej}\n")
    # Create graph and remove self-loops
    G = nx.Graph()
    G.add_edges_from(zip(s, t))
    if G.number_of_nodes()<N:
        ExpectedNodeList = [i for i in range(0, N)]
        Nodelist = list(G.nodes)
        difference = [item for item in ExpectedNodeList if item not in Nodelist]
        G.add_nodes_from(difference)
    return G, xx, yy

def R2SRGG_withgivennodepair(N, avg, beta, rg, xA, yA, xB, yB, Coorx=None, Coory=None, SaveNetworkPath=None):
    """
    Program generates Soft Random Geometric Graph on a 2d unit square
    Connection probability function is (1 + (a*d)^{b})^{-1}
    :param N: number of nodes
    :param avg: <k>
    :param beta: beta (controlling clustering)
    :param rg:  random seed
    :param Coorx: coordinates of x
    :param Coory: coordinates of y
    :param SaveNetworkPath:
    :return:
    """
    # calculate parameters
    assert beta > 2
    assert avg > 0
    assert N > 1

    R = 2.0  # manually tuned value
    alpha = (2 * N / avg * R * R) * (math.pi / (math.sin(2 * math.pi / beta) * beta))
    alpha = math.sqrt(alpha)
    s = []
    t = []

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

    xx[N-2] = xA
    yy[N-2] = yA
    xx[N-1] = xB
    yy[N-1] = yB

    # make connections
    for i in range(N):
        for j in range(i + 1, N):
            dist = math.sqrt((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2)
            assert dist > 0
            try:
                prob = 1 / (1 + math.exp(beta * math.log(alpha * dist)))
            except:
                prob = 0
            if rg.ran1() < prob:
                s.append(i)
                t.append(j)

    if SaveNetworkPath is not None:
        with open(SaveNetworkPath, "w") as file:
            for nodei, nodej in zip(s, t):
                file.write(f"{nodei}\t{nodej}\n")
    # Create graph and remove self-loops
    G = nx.Graph()
    G.add_edges_from(zip(s, t))
    if G.number_of_nodes()<N:
        ExpectedNodeList = [i for i in range(0, N)]
        Nodelist = list(G.nodes)
        difference = [item for item in ExpectedNodeList if item not in Nodelist]
        G.add_nodes_from(difference)
    return G, xx, yy


def loadSRGGandaddnode(N, filepath):
    """
    :param N: Nodenumber
    :param filepath: the txt.file that saves the network(Only edge information), the coordinates information will be saved
    in a file in the same folder called CoorED{EDn}Beta{betan}PYSimu{ST}.txt
    :return: a nx.graph
    """
    G = nx.read_edgelist(filepath, nodetype=int)
    if G.number_of_nodes() < N:
        ExpectedNodeList = [i for i in range(0, N)]
        Nodelist = list(G.nodes)
        difference = [item for item in ExpectedNodeList if item not in Nodelist]
        G.add_nodes_from(difference)
    return G


def check_realdegree_vs_expecteddegree():
    rg = RandomGenerator(-12)
    avg = 10
    repeat_size = random.randint(0,20)
    for i in range(repeat_size):
        rg.ran1()
    for network_size in [1000,10000,100000]:
        G,_,_ = R2SRGG(network_size, avg, 6, rg)
        real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
        print("networksize:", network_size)
        print("real ED:", real_avg)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rg = RandomGenerator(-12)
    for i in range(random.randint(1, 1000)):
        rg.ran1()
    # betavec = [2.2, 4, 8, 16, 32, 64, 128]
    # N_vec = [100,1000,10000]
    # ED_vec = [10,20,72]
    # betavec = [2.2, 4, 8, 16, 32, 64, 128]
    # N_vec = [100,1000,10000]
    # ED_vec = [10,20,72]
    #
    # for ED in ED_vec:
    #     for beta in betavec:
    #         for N in N_vec:
    #             G, Coorx, Coory = R2SRGG(N, ED, beta, rg)
    #             real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    #             print("inputpara:",(N,ED,beta))
    #             print("real ED:", real_avg)
    # check_realdegree_vs_expecteddegree()
    G, Coorx, Coory = R2SRGG(100, 15, 8, rg)
    real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    print("inputpara:", (100, 18,128))
    print("real ED:", real_avg)

    # G, Coorx, Coory = R2SRGG(1000, 14, 4, rg)
    # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    # print("inputpara:", (1000, 27, 128))
    # print("real ED:", real_avg)
    #
    # G, Coorx, Coory = R2SRGG(1000, 72, 4, rg)
    # real_avg = 2 * nx.number_of_edges(G) / nx.number_of_nodes(G)
    # print("inputpara:", (10000, 27, 128))
    # print("real ED:", real_avg)


    # x=  random.random()
    # y = random.random()
    # ax = random.random()
    # ay = random.random()
    # bx = random.random()
    # by = random.random()
    # print(x,y,ax,ay,bx,by)
    # a, _ = dist_to_geodesic_perpendicular_R2(x, y, ax, ay, bx, by)
    # print(a)
    # distance = point_to_line_distance_from_points(x, y, ax, ay, bx, by)
    # print(f"Point A to line BC distance: {distance}")

    # rg = RandomGenerator(-12)
    # start = time.time()
    #
    # R2SRGG_withgivennodepair(10000, 9999, 4, rg, 0.495, 0.505, 0.5, 0.5)
    # end = time.time()
    # print(end-start)

    # N = 10000
    # ED = 5
    # beta = 2.2
    # x_A = -0.005
    # y_A = 0
    # x_B = 0.005
    # y_B = 0
    # ExternalSimutime = 0
    # network_index = 0
    # x_coords = np.random.uniform(-0.5, 0.5, N)
    # y_coords = np.random.uniform(-0.5, 0.5, N)
    # x_coords[9998] = x_A
    # y_coords[9998] = y_A
    # x_coords[9999] = x_B
    # y_coords[9999] = y_B
    #
    # filefolder_name = "D:\\data\\geometric shortest path problem\\EuclideanSRGG\\EuclideanSoftRGGnetwork\\givendistance\\"
    # # Randomly generate 10 networks
    # FileNetworkCoorName = filefolder_name + "network_coordinates_N{Nn}xA{xA}yA{yA}xB{xB}yB{yB}centero.txt".format(
    #     Nn=N, xA=x_A, yA=y_A, xB=x_B, yB=y_B)
    # with open(FileNetworkCoorName, "w") as file:
    #     for data1, data2 in zip(x_coords, y_coords):
    #         file.write(f"{data1}\t{data2}\n")

