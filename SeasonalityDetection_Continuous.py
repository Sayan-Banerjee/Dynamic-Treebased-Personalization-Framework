"""
As discrete seasonality core algorithm, continuous seasonality core algorithm also expects a data of 12 rows and each row
will represent each month. Also the rows are ordered by month. Ex. : January at row 0 and December at row 11.

The core algorithm is stateless. Given each month's representation at each rows it tries to figure out which
 **ADJACENT** months are most similar by given representation and group 2 at a time and continue until breaking
  condition is reached.

Here more critical part is how do you want to represent the data to represent a month?
Do you want to use Booking trend data?
Booking Trend by what?
Do you want to include night qty and room qty in data?
Do you want to include each individual dates?

Or do you want to represent each month by sales statistics?
This actually make more sense from SHS perspective...but question is can we gather that data?
It is just 12 row, aggregated at highest level.

Irespective whatever we decide everything I can think of, have been handled.

As the name suggests, it do care about the **ADJACENCY** of 2 months before combining them.

"""

import numpy as np
from itertools import combinations
from ClusterTree_Utility import Queue
import logging
from SeasonalityDetection_Discrete import ClusterMonths_Discrete

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class GraphNode:
    def __init__(self, nodeId):
        assert isinstance(nodeId, int), "Id of each node should be of type integer."
        self.nodeId = nodeId
        self.connectedNodes = []
        self.encompassedNodes = []
        self.vector = None
        self.instances = None
        self.intraNodeDistance = 0.00001
        self.originalNodeIds = None

    def setConnected(self, connectedNodes):
        assert isinstance(connectedNodes, list), "List of GraphNode Instance(s) is expected here!"
        for gn in connectedNodes:
            assert isinstance(gn, GraphNode), "List of GraphNode Instance(s) is expected here!"
            self.connectedNodes.append(gn)

        return

    def removeConnected(self, potantialConnectedNodes):
        assert isinstance(potantialConnectedNodes, list), "List of GraphNode expected here!"
        connectedIds = [node.getID() for node in self.getConnected()]
        flag = False
        for gn in potantialConnectedNodes:
            assert isinstance(gn, GraphNode), "List of GraphNode Instance(s) is expected here!"
            if gn.getID() in connectedIds:
                self.connectedNodes.remove(gn)
                flag = True
        return flag

    def addConnected(self, newConnectedNode):

        assert isinstance(newConnectedNode, GraphNode), "An instance of GraphNode is expected here!"
        connectedIds = [node.getID() for node in self.getConnected()]
        newNodeId = newConnectedNode.getID()
        if newNodeId in connectedIds:
            raise RuntimeError("A node with same id is trying to get added twice!")

        self.connectedNodes.append(newConnectedNode)
        return

    def getConnected(self):
        return self.connectedNodes

    def getID(self):
        return self.nodeId

    def _getUnitVector(self, vector):
        eps = 0.0000001
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        magnitude = np.sqrt(np.sum(np.square(vector)))
        vector = vector / (magnitude + eps)

        return vector

    def setRepresentationVector(self, vector):
        try:
            self.vector = self._getUnitVector(vector)
        except RuntimeError:
            self.vector = vector

        return

    def getRepresentationVector(self):
        return self.vector

    def setEncompassedNodes(self, encompassedNodes):
        assert isinstance(encompassedNodes, list), "List of GraphNode Instance(s) is expected here!"
        encompassedNodes = list(set(encompassedNodes))
        for gn in encompassedNodes:
            assert isinstance(gn, GraphNode), "List of GraphNode expected here!"
            self.encompassedNodes.append(gn)

        return

    def getEncompassedNodes(self):
        return self.encompassedNodes

    def _calcIntraNodesDistance(self, originalNodes):
        vector1 = self.getRepresentationVector()
        dist_sum = 0.
        totalOriginalNodes = len(originalNodes)
        for orgNd in originalNodes:
            vector2 = orgNd.getRepresentationVector()
            dist_sum += self._euclidianDistance(vector1, vector2)

        return dist_sum / totalOriginalNodes

    def setIntraNodesDistance(self):
        # Validate self.vector is Not None
        # Validate self.encompassedNodes is Not None or Len(self.encompassedNodes) > 0
        if self.vector is None or self.encompassedNodes is None or len(self.encompassedNodes) == 0:
            self.intraNodeDistance = 0.
            self.originalNodeIds = [self.getID()]

        else:
            originalNodes = []
            queue = Queue()
            queue.enqueue(self)
            visited = []
            while not queue.isEmpty():
                en = queue.dequeue()

                if en.encompassedNodes is None or len(en.encompassedNodes) == 0:
                    originalNodes.append(en)

                else:
                    if not en.getID() in visited:
                        for e in en.encompassedNodes:
                            queue.enqueue(e)
                    else:
                        raise RuntimeError("Duplicate Node/NodeId!")

                visited.append(en.getID())

            self.originalNodeIds = [n.nodeId for n in originalNodes]
            self.intraNodeDistance = self._calcIntraNodesDistance(originalNodes)

        return

    def getIntraNodesDistance(self):
        return self.intraNodeDistance

    def _euclidianDistance(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def setInstances(self, instances):
        self.instances = instances
        return

    def getInstances(self):
        return self.instances


class Graph:
    def __init__(self, typ, nodes):
        if not typ.lower().strip() in ('clique', 'circular', 'connected'):
            raise ValueError("The type of the graph can only be 'clique' or 'circular' or 'connected'")
        self.typ = typ
        self._setUpNodes(nodes)
        return

    def _setUpNodes(self, nodes):
        assert isinstance(nodes, list), "Expected data type is 'Python List'"
        for n in nodes:
            if not isinstance(n, GraphNode):
                raise ValueError("All the nodes of the Graph should be instance of 'GraphNode'. ")
        self.nodes = nodes
        return

    def addNode(self, node):
        assert isinstance(node, GraphNode), "Instance should be of 'GraphNode'."
        self.nodes.append(node)
        return

    def _cosSim(self, a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def _euclidianDistance(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def findClusters(self, limitingCos, minMember=32, maxAcceptableClusters=5):
        cosVal_lowerLimit = np.cos(np.deg2rad(limitingCos))  # Cos 90 degree = 0, Cos 0 degree = 1
        logger.info("Cosine : {} Degree".format(limitingCos))
        logger.info("Max Acceptable No. of Clusters: {}".format(maxAcceptableClusters))
        workingNodes = {n.getID(): n for n in self.nodes}

        while True:
            if len(workingNodes) == 1:
                break

            recordKeeping = {}
            connected = {nId: tuple([n.getID() for n in workingNodes[nId].getConnected()])
                         for nId in workingNodes.keys()}

            for n in workingNodes.keys():
                n_vector = workingNodes[n].getRepresentationVector()
                cons = connected[n]
                for c in cons:
                    if tuple(sorted([n, c])) in recordKeeping.keys():
                        continue
                    else:
                        c_vector = workingNodes[c].getRepresentationVector()
                        nc_cs = self._cosSim(n_vector, c_vector)
                        recordKeeping[tuple(sorted([n, c]))] = nc_cs

            maxCosSimGrp = max(recordKeeping, key=recordKeeping.get)
            maxCosSim = recordKeeping[maxCosSimGrp]

            if maxCosSim < cosVal_lowerLimit and len(workingNodes) <= maxAcceptableClusters:  # changes made here
                membersByEachNode = {key: node.getInstances()
                                     for key, node in workingNodes.items()}

                notQualifiedMembers = {nodeID: totalMembers
                                       for nodeID, totalMembers in membersByEachNode.items()
                                       if totalMembers < minMember
                                       }

                if len(notQualifiedMembers) == 0:
                    break

                else:
                    foundWhomToMerge = False
                    if len(notQualifiedMembers) >= 2:
                        lookAndMergePairs = []
                        allNotQualifieldCombinations = sum(
                            [list(map(list, combinations(list(notQualifiedMembers.keys()), i))) for i in range(2, 3)],
                            [])
                        for ij in allNotQualifieldCombinations:
                            ij_sorted = tuple(sorted(ij))
                            if ij_sorted in recordKeeping.keys():
                                lookAndMergePairs.append(ij_sorted)
                        if len(lookAndMergePairs) > 0:
                            recordKeeping = {nodeID_ij: cosSim for nodeID_ij, cosSim in recordKeeping.items() if
                                             nodeID_ij in lookAndMergePairs}

                            maxCosSimGrp = max(recordKeeping, key=recordKeeping.get)
                            maxCosSim = recordKeeping[maxCosSimGrp]
                            foundWhomToMerge = True
                    if not foundWhomToMerge:
                        recordKeepingKeys = list(recordKeeping.keys())
                        lookAndMergePairs = []
                        for ij in recordKeepingKeys:
                            if any(i in ij for i in notQualifiedMembers):
                                lookAndMergePairs.append(ij)

                        assert len(lookAndMergePairs) > 0, "Something went wrong when merging loose nodes!"
                        recordKeeping = {nodeID_ij: cosSim for nodeID_ij, cosSim in recordKeeping.items() if
                                         nodeID_ij in lookAndMergePairs}
                        maxCosSimGrp = max(recordKeeping, key=recordKeeping.get)
                        maxCosSim = recordKeeping[maxCosSimGrp]
                        foundWhomToMerge = True

            newNodeId = (max(workingNodes.keys()) + 1)

            newWorkingNodes = {nId: node for nId, node in workingNodes.items() if nId not in maxCosSimGrp}
            newNode = GraphNode(newNodeId)
            allConnectedNodes = workingNodes[maxCosSimGrp[0]].getConnected() + workingNodes[
                maxCosSimGrp[1]].getConnected()
            allConnectedNodes = list(set([node for node in allConnectedNodes if node.getID() not in maxCosSimGrp]))
            newNode.setConnected(allConnectedNodes)
            toBeRemovedConnectionNodes = workingNodes[maxCosSimGrp[0]].getConnected() + workingNodes[
                maxCosSimGrp[1]].getConnected()
            toBeRemovedConnectionNodes = list(
                set([node for node in toBeRemovedConnectionNodes if node.getID() in maxCosSimGrp]))

            for node in allConnectedNodes:
                flag = node.removeConnected(toBeRemovedConnectionNodes)
                if flag:
                    node.addConnected(newNode)
            toBeDeletedIDs = tuple(maxCosSimGrp) + tuple([node.getID() for node in allConnectedNodes])
            for ids in toBeDeletedIDs:
                del connected[ids]
            connected[newNode.getID()] = tuple([n.getID() for n in newNode.getConnected()])
            for node in allConnectedNodes:
                connected[node.getID()] = tuple([n.getID() for n in node.getConnected()])

            newNode.setEncompassedNodes([workingNodes[i] for i in maxCosSimGrp])
            vector1 = workingNodes[maxCosSimGrp[0]].getRepresentationVector()
            instances1 = workingNodes[maxCosSimGrp[0]].getInstances()
            vector2 = workingNodes[maxCosSimGrp[1]].getRepresentationVector()
            instances2 = workingNodes[maxCosSimGrp[1]].getInstances()
            totalInstances = instances1 + instances2
            frac1 = instances1 / totalInstances
            frac2 = instances2 / totalInstances
            newNode.setRepresentationVector((vector1 / frac1) + (vector2 / frac2))
            newNode.setInstances(totalInstances)
            newNode.setIntraNodesDistance()
            newWorkingNodes[newNodeId] = newNode
            self.addNode(newNode)
            workingNodes = newWorkingNodes

        totalNodes = len(workingNodes)

        interClusterDistance = 0.
        intraClusterDistance = 0.
        wrkNodeIds = list(workingNodes.keys())
        if totalNodes > 1:
            allCombos = sum([list(map(list, combinations(wrkNodeIds, i))) for i in range(2, 3)], [])
            for i, j in allCombos:
                dist = self._euclidianDistance(workingNodes[i].getRepresentationVector(),
                                               workingNodes[j].getRepresentationVector())
                interClusterDistance += dist

        interClusterDistance = (interClusterDistance / totalNodes) + 1.
        logger.info("Inter-Cluster Distance: {}".format(interClusterDistance))

        for ii in wrkNodeIds:
            intraClusterDistance += workingNodes[ii].getIntraNodesDistance()
        intraClusterDistance = intraClusterDistance / totalNodes
        logger.info("Intra-Cluster Distance: {}".format(intraClusterDistance))
        purity = intraClusterDistance / interClusterDistance
        logger.info("Inv-Purity = (Intra-Cluster Distance/Inter-Cluster Distance) : {}".format(intraClusterDistance))
        clusters = {}
        i = 1
        for c in workingNodes.keys():
            name = 'Cluster-{}'.format(i)
            val = workingNodes[c].originalNodeIds if workingNodes[c].originalNodeIds is not None else workingNodes[
                c].getID()
            clusters[name] = val
            i += 1

        return purity, clusters


def buildTheGraph(ds_EachRowADataPoint, instanceCountByEachDataPoint=None, graphType='circular'):
    # ds_EachRowADataPoint ==> Data Source, of which each row is a representation of a data
    # point and id of a data point
    # is the index of the data point. Starts with 1.
    originalNodes = []
    connectedNodes = []
    for i in range(1, len(ds_EachRowADataPoint) + 1):
        row = ds_EachRowADataPoint.loc[i, ].values
        node = GraphNode(nodeId=i)
        node.setRepresentationVector(row)
        if instanceCountByEachDataPoint is not None:
            node.setInstances(instanceCountByEachDataPoint[i])
        else:
            node.setInstances(1)

        left = i - 1 if i > 1 else len(ds_EachRowADataPoint)
        right = i + 1 if i < len(ds_EachRowADataPoint) else 1
        connectedNodes.append([left, right])
        originalNodes.append(node)

    for ii in range(len(originalNodes)):
        node = originalNodes[ii]
        leftPos = connectedNodes[ii][0] - 1
        rightPos = connectedNodes[ii][1] - 1
        node.setConnected([originalNodes[leftPos], originalNodes[rightPos]])

    graph = Graph(graphType, originalNodes)

    return graph


def validateDiscreteResult(clusters):
    flag = True
    for c in clusters.keys():
        val = clusters[c].copy()
        if 1 in val and 12 in val:
            val.remove(12)
            val.append(0)
            val = sorted(val)

        val = [val[i] - val[i - 1] for i in range(1, len(val))]
        d = max(val)
        if d == 1:
            continue
        else:
            flag = False
            break
    if flag:
        return clusters
    else:
        return None


def ClusterMonths_Continuous(ds_EachRowADataPoint,
                             instanceCountByEachDataPoint=None, cosDegrees=None,
                             acceptableNumberOfClusters=None):
    clusters_D = ClusterMonths_Discrete(ds_EachRowADataPoint)
    clusters_D = validateDiscreteResult(clusters_D)
    if clusters_D is not None:
        return clusters_D

    if cosDegrees is None:
        cosDegrees = list(np.linspace(0.5, 30, 60))
    if acceptableNumberOfClusters is None:
        acceptableNumberOfClusters = [4, 5, 6]

    purityList = {}
    clusterList = {}
    for cos in cosDegrees:
        for maxAllowedClusters in acceptableNumberOfClusters:
            # Circle graph prevents merging 2 nodes which are not adjacent by its natural sense.
            # Ex.: January is connected to December and February but NOT with any other month.
            # So, even if January has very similar booking trend with May, this algorithm will NOT
            # merge them to create a new node like normal Hierarchical clustering.
            graph = buildTheGraph(ds_EachRowADataPoint, instanceCountByEachDataPoint, 'circular')
            purity, clusters = graph.findClusters(limitingCos=cos, maxAcceptableClusters=maxAllowedClusters)
            purityList[(cos, maxAllowedClusters)] = purity
            clusterList[(cos, maxAllowedClusters)] = clusters

    mostPureAt = min(purityList, key=purityList.get)
    logger.info("Seasonality Detection Among Continuous Months ---- Most Pure At : {}".format(mostPureAt))
    graph = buildTheGraph(ds_EachRowADataPoint, instanceCountByEachDataPoint, 'circular')
    _, clusters = graph.findClusters(limitingCos=mostPureAt[0], maxAcceptableClusters=mostPureAt[1])

    return clusters
