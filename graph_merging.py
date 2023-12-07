import sys
import os
import numpy as np
import math
import numpy as np
import pandas
import networkx as nx
from networkx.algorithms import graph_edit_distance

def dbscan(D, eps, MinPts):
    
    labels = [0]*len(D)
   
    C = 0

    for P in range(0, len(D)):

        if not (labels[P] == 0):
           continue
        
        NeighborPts = region_query(D, P, eps)
        
        if len(NeighborPts) < MinPts:
            labels[P] = -1
  
        else: 
           C += 1
           grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts)

    return labels


def grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts):

    labels[P] = C
    
    i = 0
    while i < len(NeighborPts):    
        
        Pn = NeighborPts[i]
       
        if labels[Pn] == -1:
           labels[Pn] = C
        
        elif labels[Pn] == 0:
            labels[Pn] = C
            
            PnNeighborPts = region_query(D, Pn, eps)

            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts

        i += 1        


def region_query(D, P, eps):
 
    neighbors = []
    for Pn in range(0, len(D)):
        G1 = D[P]
        G2 = D[Pn]
        distance = graph_edit_distance(G1, G2)
        max_distance = max(nx.number_of_nodes(G1), nx.number_of_nodes(G2))
        similarity_score = (max_distance - distance) / max_distance
        if similarity_score < eps:
           neighbors.append(Pn)
            
    return neighbors