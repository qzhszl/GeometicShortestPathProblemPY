# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 6-6-2024
"""
import math
import random

import networkx as nx

from SphericalSoftRandomGeomtricGraph import RandomGenerator


def RandomGeometricGraph(N, avg, rg, radius = None, Coorx=None, Coory=None, SaveNetworkPath=None):
    """
        %%***********************************************************************%
        %*             Random Geometric Graph Generator on 2-d euclidean space  *%
        %*             Generates euclidean Random Geometric Graphs             *%
        %*                                                                      *%
        %*                                                                      *%
        %* Author: Zhihao Qiu                                                   *%
        %* Date: 6/06/2024                                                     *%
        %************************************************************************%
        %
        %************************************************************************%
        %
        % Usage: G, x, y     = RandomGeometricGraph(N, avg, beta, rg, Coortheta, Coorphi, "filename.txt")
        %
        % Inputs:
        %           N                   - Number of nodes in the graph
        %           avg                 - Expected degree
        %           rg                  - random number seed generator
        %           Coorx/Coory   - Optional positions for nodes in the graph
        %           SaveNetworkPath     - "savefilepath.txt"
        %
        %
        % Outputs:
        %
        %           G                   - Graph object for the random geometric graph
        %           x, y      - (x, y)coordinates of all each point in the graph
        %--------------------------------------------------------------------------
        """
    assert N > 1
    assert avg > 0

    s = []
    t = []
    if radius is None:
        radius = math.sqrt(avg/((N-1)*math.pi))

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
            if dist < radius:
                s.append(i)
                t.append(j)

    if SaveNetworkPath is not None:
        with open(SaveNetworkPath, "w") as file:
            for nodei, nodej in zip(s, t):
                file.write(f"{nodei}\t{nodej}\n")
    # Create graph and remove self-loops
    G = nx.Graph()
    G.add_edges_from(zip(s, t))
    if G.number_of_nodes() < N:
        ExpectedNodeList = [i for i in range(0, N)]
        Nodelist = list(G.nodes)
        difference = [item for item in ExpectedNodeList if item not in Nodelist]
        G.add_nodes_from(difference)
    return G, xx, yy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # rg = RandomGenerator(-12)  # Seed initialization
    # for _ in range(random.randint(0, 100)):
    #     rg.ran1()
    # # radius = 0.012616293440543984
    # G, xx, yy = RandomGeometricGraph(10000,5,rg)
    # print(nx.number_of_nodes(G))
    # print(nx.number_of_edges(G))
    # print(2*nx.number_of_edges(G)/nx.number_of_nodes(G))
    radius = math.sqrt(10/((10000-1)*math.pi))
    print(radius)