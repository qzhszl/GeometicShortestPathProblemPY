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


def find_nonzero_indices(lst):
    # 创建一个空列表来存储非0元素的索引
    nonzero_indices = []

    # 遍历列表，使用enumerate获取元素及其索引
    for index, element in enumerate(lst):
        # 如果元素非0，则添加其索引到nonzero_indices列表中
        if element != 0:
            nonzero_indices.append(index)

    return nonzero_indices

def find_nonnan_indices(lst):
    # 创建一个空列表来存储非0元素的索引
    nonzero_indices = []
    # 遍历列表，使用enumerate获取元素及其索引
    for index, element in enumerate(lst):
        # 如果元素非0，则添加其索引到nonzero_indices列表中
        if not np.isnan(element):
            nonzero_indices.append(index)
    return nonzero_indices

def all_shortest_path_node(G, nodei, nodej):
    """
        :return: All shortest path nodes except for nodei and nodej
    """
    shortest_paths = nx.all_shortest_paths(G, nodei, nodej)
    PNodeList = set()  # Use a set to keep unique nodes
    count = 0
    for path in shortest_paths:
        PNodeList.update(path)
        count += 1
        if count > 1000000:
            break
    PNodeList.discard(nodei)
    PNodeList.discard(nodej)
    PNodeList = list(PNodeList)
    return PNodeList

def find_top_n_values(arr, N):
    # 找到最大的N个数的索引（在数组中的索引）
    indices = np.argpartition(arr, -N)[-N:]
    # 根据这些索引找到对应的值
    top_values = arr[indices]
    # 对这些值按降序排序
    sorted_indices = indices[np.argsort(-top_values)]
    # 按顺序返回最大的N个数的值和对应的索引
    top_values_sorted = arr[sorted_indices]

    return top_values_sorted, sorted_indices

def testallspnodes(G,source, target):
    pred = nx.predecessor(G, source)
    if target not in pred:
        raise nx.NetworkXNoPath()
    stack = [[target, 0]]
    top = 0
    while top >= 0:
        node, i = stack[top]
        if node == source:
            yield [p for p, n in reversed(stack[:top + 1])]
        if len(pred[node]) > i:
            top += 1
            if top == len(stack):
                stack.append([pred[node][i], 0])
            else:
                stack[top] = [pred[node][i], 0]
        else:
            stack[top - 1][1] += 1
            top -= 1

def find_all_connected_node_pairs(G):
    connected_node_pairs = set()
    components = list(nx.connected_components(G))
    connected_component = [s for s in components if len(s) > 1]
    for compo in connected_component:
        unique_connected_pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(compo, 2))
        connected_node_pairs = set.union(unique_connected_pairs,connected_node_pairs)
    return connected_node_pairs

def find_k_connected_node_pairs(G,k):
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    nodes = list(largest_component)
    if len(nodes)*(len(nodes)-1)/2>k:
        # 已生成的节点对集合
        generated_pairs = set()
        # 结果列表
        unique_connected_pairs = []
        while len(unique_connected_pairs) < k:
            # 随机选择两个不同的节点
            node1, node2 = random.sample(range(G.number_of_nodes()), 2)
            # 生成的节点对
            node_pair = tuple(sorted((node1, node2)))  # 确保 (node1, node2) 和 (node2, node1) 被视为相同的对
            # 检查节点对的连通性及是否已生成过
            if node_pair not in generated_pairs and nx.has_path(G, node1, node2):
                generated_pairs.add(node_pair)
                unique_connected_pairs.append(node_pair)
    else:
        unique_connected_pairs = find_all_connected_node_pairs(G)
    return unique_connected_pairs


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
    # G.add_edges_from([(0, 1),(0,2),(2,3), (1,3),(4, 3),(3,5),(4,6),(5,6)])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (5, 4)])
    G.add_node(7)
    connected_component = find_all_connected_node_pairs(G)

    # components = list(nx.connected_components(G))
    # connected_component = [s for s in components if len(s)>1]
    # plt.figure()
    # nx.draw(G)
    # plt.show()
    c = find_k_connected_node_pairs(G, 100)
    print(c)



    # nodei = 0
    # nodej = 4
    # source = 0
    # target = nodej
    #
    # pathtest = testallspnodes(G, source, target)
    # for i in pathtest:
    #     print(i)
    # pred = nx.predecessor(G, source)
    # if target not in pred:
    #     raise nx.NetworkXNoPath()
    # stack = [[target, 0]]
    # top = 0
    # while top >= 0:
    #     node, i = stack[top]
    #     if node == source:
    #         res = [p for p, n in reversed(stack[:top + 1])]
    #     if len(pred[node]) > i:
    #         top += 1
    #         if top == len(stack):
    #             stack.append([pred[node][i], 0])
    #         else:
    #             stack[top] = [pred[node][i], 0]
    #     else:
    #         stack[top - 1][1] += 1
    #         top -= 1
    # print(res)
    # shortest_paths = nx.all_shortest_paths(G, nodei, nodej)
    # # testa = shortest_paths.gi_frame.f_locals["pred"]
    # # print(testa)
    # Pnodelist_test = set()
    # for node in testa.values():
    #     Pnodelist_test.update(node)
    # Pnodelist_test.discard(nodei)
    # print(list(Pnodelist_test))


    # PNodeList = set()  # Use a set to keep unique nodes
    # count = 0
    # for path in shortest_paths:
    #     print(path)
    #     PNodeList.update(path)
    #     count += 1
    #     if count > 1000000:
    #         break
    # PNodeList.discard(nodei)
    # PNodeList.discard(nodej)
    # PNodeList = list(PNodeList)
    #
    # print("Pnodelist:", PNodeList)


    # d1=nx.shortest_path_length(G, 0, 0)
    # d2 = nx.shortest_path_length(G, 0, 3)
    # print(min(d1,d2))
    # # 获取网络中的所有连通分量
    # components = list(nx.connected_components(G))
    # # 找到最大的连通分量
    # largest_component = max(components, key=len)
    # # 从最大连通分量中获取节点列表
    # nodes = list(largest_component)
    # # 创建所有可能的节点对
    # unique_pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(nodes, 2))
    # # 从唯一节点对中随机选择 100 个
    # random_pairs = random.sample(sorted(unique_pairs), 2)
    # print(random_pairs)



