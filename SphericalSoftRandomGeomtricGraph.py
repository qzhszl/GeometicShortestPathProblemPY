# -*- coding UTF-8 -*-
"""
@Project: Geometric shortest path problem
@Author: Zhihao Qiu
@Date: 2024/4/29
"""
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import sys

# Constants
# Functions
# Constants
IA = 16807
IM = 2147483647
AM = 1.0 / IM
IQ = 127773
IR = 2836
NTAB = 32
NDIV = 1 + (IM - 1) // NTAB
EPS = 1.2e-7
RNMX = 1.0 - EPS


class RandomGenerator:
    def __init__(self, seed):
        self.idum = seed
        self.iv = [0] * NTAB
        self.iy = 0
        if self.idum <= 0 or self.iy == 0:
            if -self.idum < 1:
                self.idum = 1  # Be sure to prevent idum = 0
            else:
                self.idum = -self.idum
            for j in range(NTAB + 7, -1, -1):
                k = self.idum // IQ
                self.idum = IA * (self.idum - k * IQ) - IR * k
                if self.idum < 0:
                    self.idum += IM
                if j < NTAB:
                    self.iv[j] = self.idum
            self.iy = self.iv[0]

    def ran1(self):
        k = self.idum // IQ
        self.idum = IA * (self.idum - k * IQ) - IR * k  # Compute without overflows using Schrage's method
        if self.idum < 0:
            self.idum += IM

        j = self.iy // NDIV  # Get index for shuffle table
        self.iy = self.iv[j]  # Output previously stored value and refill the shuffle table
        self.iv[j] = self.idum

        temp = AM * self.iy
        return min(temp, RNMX)


def sin_generator(rg):  # Input: Random generator
    return math.acos(1 - 2 * rg.ran1())


def distS2(angle1i, angle2i, angle1j, angle2j):
    dist = math.cos(angle1i) * math.cos(angle1j)
    dist += math.sin(angle1i) * math.sin(angle1j) * math.cos(angle2i) * math.cos(angle2j)
    dist += math.sin(angle1i) * math.sin(angle1j) * math.sin(angle2i) * math.sin(angle2j)
    return math.acos(dist)


def dist_to_geodesic_S2(angle1C, angle2C, angle1A, angle2A, angle1B, angle2B):
    # Finds shortest distance from C to arc AB on a unit sphere.
    AB = distS2(angle1A, angle2A, angle1B, angle2B)
    assert 0.00000001 < AB < math.pi
    AC = distS2(angle1A, angle2A, angle1C, angle2C)
    assert 0.00000001 < AC < math.pi
    BC = distS2(angle1B, angle2B, angle1C, angle2C)
    assert 0.00000001 < BC < math.pi
    alpha = math.acos((math.cos(BC) - math.cos(AB) * math.cos(AC)) / (math.sin(AB) * math.sin(AC)))
    alpha2 = (math.cos(AC) - math.cos(AB) * math.cos(BC)) / (math.sin(AB) * math.sin(BC))
    alpha2 = math.acos(alpha2)
    if alpha < math.pi / 2 and alpha2 < math.pi / 2:
        dist = math.sin(AC) * math.sin(alpha)
        dist = math.asin(dist)
    elif alpha > alpha2:
        dist = distS2(angle1C, angle2C, angle1A, angle2A)
    else:
        dist = distS2(angle1C, angle2C, angle1B, angle2B)
    assert 0 <= dist < math.pi
    return dist


def SphericalSoftRGG(N, avg, beta, rg, Coortheta=None,Coorphi=None):
    assert N >1
    assert beta>2
    assert avg>0

    s = []
    t = []

    # Calculate the alpha parameter
    alpha = math.sqrt((N * math.pi) / (2 * avg * beta * math.sin(2 * math.pi / beta)))

    # Assign coordinates
    if Coortheta is not None and Coorphi is not None:
        angle1 = Coortheta
        angle2 = Coorphi
    else:
        angle1 = []
        angle2 = []
        for i in range(N):
            sinangle = sin_generator(rg)
            angle1.append(sinangle)
            rannum = rg.ran1()
            angle2.append(2 * math.pi * rannum)

    # Make connections
    for i in range(N):
        for j in range(i + 1, N):
            dist = distS2(angle1[i], angle2[i], angle1[j], angle2[j])
            prob = 1 / (1 + math.exp(beta * math.log(alpha * dist)))
            rannum = rg.ran1()
            if rannum < prob:
                s.append(i)
                t.append(j)
    # Create graph and remove self-loops
    G = nx.Graph()
    G.add_edges_from(zip(s, t))
    return G, angle1, angle2


if __name__ == "__main__":
    # dist = dist_to_geodesic_S2(math.pi/4,0,math.pi/2,math.pi/2,math.pi/2,math.pi)
    # print(dist)
    rg = RandomGenerator(-12)  # Seed initialization
    N= 100
    avg = 5
    beta = 3.5
    G, Coortheta, Coorphi = SphericalSoftRGG(N, avg, beta, rg, Coortheta=None, Coorphi=None)
    print(Coortheta)
    print(Coorphi)

    # filename = "D:\\data\\geometric shortest path problem\\SSRGG\\PRAUC\\testnetworkNode{NodeNum}.txt".format(NodeNum = N)
    # print(filename)
