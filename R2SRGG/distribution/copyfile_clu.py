# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2-12-2024
"""


import os
import shutil
import numpy as np

# 定义输入值范围
N_values = [100, 200, 300]
kvec = np.arange(2, 6.1, 0.2)
kvec = [round(a, 1) for a in kvec]
betavec = [2.1, 4, 8, 16, 32, 64, 128]
beta_values = [0.1, 0.2, 0.3]

# 源文件路径模板
source_template = "/home/zqiu1/GSPP_SRGG_Dev/NetworkSRGG/network_N{Nn}ED{EDn}Beta{betan}.txt"

# 目标文件夹
destination_folder = "/work/zqiu1"

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 循环遍历输入值，生成文件名并复制文件
for N in [10000]:
    for ED in kvec:
        for beta in beta_values:
            source_file = source_template.format(Nn=N, EDn=ED, betan=beta)
            destination_file = os.path.join(destination_folder, os.path.basename(source_file))
            try:
                shutil.copy(source_file, destination_file)
                print(f"Copied: {source_file} -> {destination_file}")
            except FileNotFoundError:
                print(f"File not found: {source_file}")
            except PermissionError:
                print(f"Permission denied: {source_file}")