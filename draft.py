import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
G.add_edges_from([(0, 2),(1,3)])
if G.number_of_nodes() < 7:
    ExpectedNodeList = [i for i in range(0, 7)]
    Nodelist = list(G.nodes)
    difference = [item for item in ExpectedNodeList if item not in Nodelist]
    G.add_nodes_from(difference)
plt.figure()
nx.draw(G)
plt.show()

print(nx.current_flow_betweenness_centrality(G, normalized=False))