import pandas as pd
import numpy as np
import logging
import scipy.cluster.hierarchy as shc

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def ClusterMonths_Discrete(data, plot=False):
    
    tree = shc.linkage(data, method='ward')
    if plot:
        dend = shc.dendrogram(tree)
        
    vertices = {i:tuple([i]) for i in range(12)}
    i=0
    while len(vertices)>2:
        m = max(vertices.keys())
        vx1 = tree[i, 0]
        vx2 = tree[i, 1]
        distance = tree[i, 2]
        if len(vertices)==3 and distance>=1: 
            break
        newVxKey = m + 1
        vx1Val = vertices.get(int(vx1))
        vx2Val = vertices.get(int(vx2))
        newVxVal = vx1Val + vx2Val
        del vertices[vx1]
        del vertices[vx2]
        vertices[newVxKey] = newVxVal
        i+=1
    
    l = {}
    for key in vertices.keys():
        l[key] = len(vertices[key])
    
    if min(l.items(), key=lambda x: x[1])[1] == 1:
        vertices = mergeWithNextMinimum(vertices, l)
        
    clusters = {}
    ii=0
    for key in vertices.keys():
        ii+=1
        months = [i+1 for i in vertices[key]]
        cKey = '{}-{}'.format('Cluster', ii)
        clusters[cKey] = months
        
    return clusters

def mergeWithNextMinimum(vertices, l):
    l = sorted(l.items(), key=lambda x: x[1])
    vx1 = l[0][0]
    vx2 = l[1][0]
    m = max(vertices.keys())
    newVxKey = m + 1
    vx1Val = vertices.get(int(vx1))
    vx2Val = vertices.get(int(vx2))
    newVxVal = vx1Val + vx2Val
    del vertices[vx1]
    del vertices[vx2]
    vertices[newVxKey] = newVxVal
    return vertices

