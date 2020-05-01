"""
Discrete seasonality core algorithm expects a data of 12 rows and each row will represent each month.
Also the rows are ordered by month. Ex. : January at row 0 and December at row 11.

The core algorithm is stateless. Given each month's representation at each rows it tries to figure out which months
 are most similar by given representation and group 2 at a time and continue until breaking condition is reached.

Here more critical part is how do you want to represent the data to represent a month?
Do you want to use Booking trend data?
Booking Trend by what?
Do you want to include night qty and room qty in data?
Do you want to include each individual dates?

Or do you want to represent each month by sales statistics?
This actually make more sense from SHS perspective...but question is can we gather that data?
It is just 12 row, aggregated at highest level.

Irrespective whatever we decide everything I can think of, have been handled.

As the name suggests, it doesn't care about the "adjacency" of 2 months befre combining them.
"""

import logging
import scipy.cluster.hierarchy as shc

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def ClusterMonths_Discrete(data, plot=False):
    tree = shc.linkage(data, method='ward')
    if plot:
        dend = shc.dendrogram(tree)

    vertices = {i: tuple([i]) for i in range(12)}
    i = 0
    while len(vertices) > 2:
        m = max(vertices.keys())
        vx1 = tree[i, 0]
        vx2 = tree[i, 1]
        distance = tree[i, 2]
        if len(vertices) == 3 and distance >= 1:
            break
        newVxKey = m + 1
        vx1Val = vertices.get(int(vx1))
        vx2Val = vertices.get(int(vx2))
        newVxVal = vx1Val + vx2Val
        del vertices[vx1]
        del vertices[vx2]
        vertices[newVxKey] = newVxVal
        i += 1

    groups = {}
    for key in vertices.keys():
        groups[key] = len(vertices[key])

    if min(groups.items(), key=lambda x: x[1])[1] == 1:
        vertices = mergeWithNextMinimum(vertices, groups)

    clusters = {}
    ii = 0
    for key in vertices.keys():
        ii += 1
        months = [i + 1 for i in vertices[key]]
        cKey = '{}-{}'.format('Cluster', ii)
        clusters[cKey] = months

    return clusters


def mergeWithNextMinimum(vertices, groups):
    groups = sorted(groups.items(), key=lambda x: x[1])
    vx1 = groups[0][0]
    vx2 = groups[1][0]
    m = max(vertices.keys())
    newVxKey = m + 1
    vx1Val = vertices.get(int(vx1))
    vx2Val = vertices.get(int(vx2))
    newVxVal = vx1Val + vx2Val
    del vertices[vx1]
    del vertices[vx2]
    vertices[newVxKey] = newVxVal
    return vertices
