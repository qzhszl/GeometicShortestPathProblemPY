# This is a sample Python script.
import itertools
import random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import SphericalSoftRandomGeomtricGraph

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    G = nx.fast_gnp_random_graph(100,0.1)

def GenerateGraph():
    # Use a breakpoint in the code line below to debug your script.
    G = nx.fast_gnp_random_graph(100, 0.1)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


def IOforNormalFile():
    Coortheta =[1,2,3]
    Coorphi = [4,5,6]
    with open("D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\output.txt", "w") as file:
        file.write("# Name\tAge\n")  # 使用制表符分隔列
        # 写入数据
        for name, age in zip(Coortheta, Coorphi):
            file.write(f"{name}\t{age}\n")
    Coorphi = []
    with open("D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\output.txt", "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split("\t")  # 使用制表符分割
            Coorphi.append(float(data[1]))
    FileASPName = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\output.txt"
    np.savetxt(FileASPName, Coorphi,fmt="%i")
    data = np.loadtxt(FileASPName, dtype=int)


def IOforGraph():
    G = nx.Graph()
    nx.write_edgelist(G,"D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\testnetwork.txt")
    G = nx.read_edgelist("D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\testnetwork.txt",nodetype=int)
    print(G.edges)


def sprintffilename():
    N=1
    filename = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\testnetworkNode{NodeNum}.txt".format(NodeNum = N)
    print(filename)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # print_pi()

    # graph = nx.convert_node_labels_to_integers(graph, first_label=1)

    # # nx.write_edgelist(G,"D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\testnetwork.txt")
    # G = nx.read_edgelist("D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\testnetwork.txt",delimiter=",",nodetype=int)
    # print(G.edges)

    # G = nx.fast_gnp_random_graph(20, 0.1)
    G = nx.Graph()
    G.add_edges_from([(1, 0), (1,3),(2, 3),(4,5)])
    # 获取网络中的所有连通分量
    components = list(nx.connected_components(G))
    # 找到最大的连通分量
    largest_component = max(components, key=len)
    # 从最大连通分量中获取节点列表
    nodes = list(largest_component)
    # 创建所有可能的节点对
    unique_pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(nodes, 2))
    # 从唯一节点对中随机选择 100 个
    random_pairs = random.sample(sorted(unique_pairs), 2)
    print(random_pairs)