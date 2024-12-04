# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2-12-2024
copy, paste network into /work/zqiu1
"""


import os
import shutil
import numpy as np
import networkx as nx
from R2SRGG import R2SRGG, distR2, dist_to_geodesic_R2, loadSRGGandaddnode

def check_where_is_the_problem(source_file, destination_file):
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



if __name__ == '__main__':
    kvec = np.arange(2, 6.1, 0.2)
    kvec = [round(a, 1) for a in kvec]
    beta_values = [2.2, 4, 8, 16, 32, 64, 128]

    # 源文件路径模板
    source_folder = "/home/zqiu1/GSPP_SRGG_Dev/NetworkSRGG/"
    # # 目标文件夹
    destination_folder = "/work/zqiu1/"
    file_template = "network_N{Nn}ED{EDn}Beta{betan}.txt"
    # 确保目标文件夹存在
    # os.makedirs(destination_folder, exist_ok=True)
    # FileNetworkName = "/home/zqiu1/GSPP_SRGG_Dev/NetworkSRGG/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #                 Nn=N, EDn=ED, betan=beta)
    # G = loadSRGGandaddnode(N, FileNetworkName)
    # # load coordinates with noise
    # Coorx = []
    # Coory = []
    #
    # FileNetworkCoorName = "/home/zqiu1/GSPP_SRGG_Dev/NetworkSRGG/network_coordinates_N{Nn}ED{EDn}Beta{betan}.txt".format(
    #     Nn=N, EDn=ED, betan=beta)
    # with open(FileNetworkCoorName, "r") as file:
    #     for line in file:
    #         if line.startswith("#"):
    #             continue
    #         data = line.strip().split("\t")  # 使用制表符分割
    #         Coorx.append(float(data[0]))
    #         Coory.append(float(data[1]))

    # 循环遍历输入值，生成文件名并复制文件
    for N in [10000]:
        for ED in [kvec[0]]:
            for beta in [beta_values[0]]:
                source_file = source_folder+file_template.format(Nn=N, EDn=ED, betan=beta)
                destination_file = os.path.join(destination_folder, os.path.basename(source_file))
                check_where_is_the_problem(source_file,destination_file)

                FileNetworkName = "/work/zqiu1/network_N{Nn}ED{EDn}Beta{betan}.txt".format(
                    Nn=N, EDn=ED, betan=beta)
                G = loadSRGGandaddnode(N, FileNetworkName)
                print(nx.number_of_edges(G))

                # shutil.copy(source_file, destination_file)
                # try:
                #     shutil.copy(source_file, destination_file)
                #     print(f"Copied: {source_file} -> {destination_file}")
                # except FileNotFoundError:
                #     print(f"File not found: {source_file}")
                # except PermissionError:
                #     print(f"Permission denied: {source_file}")