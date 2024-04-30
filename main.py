# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import networkx as nx
import matplotlib.pyplot as plt
from SphericalSoftRandomGeomtricGraph import print_pi

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
    print_hi('PyCharm')
    print_pi()

    # graph = nx.convert_node_labels_to_integers(graph, first_label=1)

    # # nx.write_edgelist(G,"D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\testnetwork.txt")
    # G = nx.read_edgelist("D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\testnetwork.txt",delimiter=",",nodetype=int)
    # print(G.edges)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
