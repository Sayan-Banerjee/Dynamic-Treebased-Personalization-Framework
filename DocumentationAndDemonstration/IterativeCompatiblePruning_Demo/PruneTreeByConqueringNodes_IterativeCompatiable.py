import copy
import logging
import scipy.cluster.hierarchy as shc
import pandas as pd
import numpy as np
from itertools import product, combinations
from CLTreeModules_IterativeCompatible import *
from CLTree_Utility import *
from ReadWritePickleFile import *
from ClusterTree_Utility import Queue

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class PruneTreeByConqueringNodes(object):
    def __init__(self, prefixString, cltree, conquerData, useSilhouette, tmpFilesPath='./'):
        self.cltree = cltree
        self.root = self.cltree.getRoot()
        self.version = None
        if isinstance(prefixString, str):
            self.prefixString = prefixString
        else:
            self.prefixString = str(prefixString)
        assert isinstance(conquerData, pd.DataFrame), "'conquerData' has to be of type pandas.DataFrame!"
        self.conquerData = conquerData
        self.useSilhouette = useSilhouette
        while True:
            tmpRunningId = getRandomStringasId()
            tmpFilesPath = tmpFilesPath + str(tmpRunningId) + "/"
            if doThisPathExist(tmpFilesPath):
                continue
            else:
                createPath(tmpFilesPath)
                self.tmpFilesPath = tmpFilesPath
                break

    def prune(self, min_y, gradientTolerance = 0.1, searchMode=True):
        self.gradientTolerance = gradientTolerance
        self.min_y = max(1, min_y)
        self.maxSoFar = -1
        self.version = 3.0
        self.totalDataInstancesbyEachGroup = None
        self.finalRepresentationVectors = None
        self.finalRepresentationVectorsToCluster = None
        self.originalRepresentationVectors = None
        self.intializeTouchedNodes(self.root)
        self.originalRepresentationVectors = self.initializeRepresentationVectors(self.root)


        totalDataInstancesbyEachGroup, finalRepresentationVectors, finalRepresentationVectorsToCluster = \
                        self.pruningTreeByMergingRepresentationVectors(self.originalRepresentationVectors, self.min_y)

        if totalDataInstancesbyEachGroup is None:
            return None

        self.totalDataInstancesbyEachGroup = totalDataInstancesbyEachGroup
        self.finalRepresentationVectors = finalRepresentationVectors
        self.finalRepresentationVectorsToCluster = finalRepresentationVectorsToCluster
        self.leafPruning(self.root)

        result = self.clusterStatistics(copy.deepcopy(self.originalRepresentationVectors),
                                        copy.deepcopy(self.finalRepresentationVectors),
                                        copy.deepcopy(self.finalRepresentationVectorsToCluster),
                                        copy.deepcopy(self.totalDataInstancesbyEachGroup))

        if not searchMode:
            self.pruningRedundantNodes(self.cltree.root)
            result2 = self.clusterStatistics(self.originalRepresentationVectors,
                                             self.finalRepresentationVectors,
                                             self.finalRepresentationVectorsToCluster,
                                             self.totalDataInstancesbyEachGroup)

            logger.info("Search Mode: False.\
        Note: The underlying Tree structure got modified in the process of pruning to make the search space optimized!\
        'min_y' = {}, 'result' = \n{}".format(min_y, result2))
            return result2

        else:
            logger.info("Searching for 'min_y'. Current 'min_y' = {}, 'result' = \n{}".format(min_y, result))
            return result

    def identifyingTouchedRegions(self, root):
        # Get back a preorder traversal list of all leaf nodes.
        prunePreOrderedList = self.preOrderTraversal(root, list())
        mergeList = []
        for i in range(len(prunePreOrderedList)):
            node_i = self.cltree.nodes[prunePreOrderedList[i]]
            data_i = node_i.getDatasetSummary()
            attributes_i = data_i.getAttributes()

            for j in range(i + 1, len(prunePreOrderedList)):
                node_j = self.cltree.nodes[prunePreOrderedList[j]]
                data_j = node_j.getDatasetSummary()
                attr_ij = None
                flag_ij = False
                for attr in attributes_i:

                    min_i = data_i.get_min(attr)
                    max_i = data_i.get_max(attr)
                    min_j = data_j.get_min(attr)
                    max_j = data_j.get_max(attr)
                    if min_i == max_j or max_i == min_j:
                        flag_ij = True
                        attr_ij = attr
                        break

                if flag_ij:
                    otherAttrs_i = [attr for attr in attributes_i if attr != attr_ij]

                    for attr in otherAttrs_i:
                        min_i = data_i.get_min(attr)
                        max_i = data_i.get_max(attr)
                        min_j = data_j.get_min(attr)
                        max_j = data_j.get_max(attr)
                        if (min_i < max_j and max_i > min_j) or (min_j < max_i and max_j > min_i):
                            continue
                        else:
                            flag_ij = False
                            break
                    if flag_ij:
                        logger.info("Adding to merge list: {}".format((node_i.getId(), node_j.getId())))
                        mergeList.append((prunePreOrderedList[i], prunePreOrderedList[j]))

        touchingNodes = dict()
        logger.info("Printing MergeList: {}".format(mergeList))
        if len(mergeList) > 0:
            touchingNodes = self.normalizingMergeList(mergeList)
        logger.info("Printing Touching Nodes: {}".format(touchingNodes))
        return touchingNodes

    def normalizingMergeList(self, mergeList):
        touchingNodes = dict()
        for nodeId_i, nodeId_j in mergeList:
            list_i = touchingNodes.get(nodeId_i, set())
            list_j = touchingNodes.get(nodeId_j, set())
            list_i.add(nodeId_j)
            list_j.add(nodeId_i)
            touchingNodes[nodeId_i] = list_i
            touchingNodes[nodeId_j] = list_j

        for nodeId in touchingNodes.keys():
            queue = Queue()
            visited = set(touchingNodes[nodeId])
            for i in visited:
                queue.enqueue(i)
            while not queue.isEmpty():
                n = queue.dequeue()
                for node in touchingNodes[n]:
                    if node not in visited:
                        queue.enqueue(node)
                        visited.add(node)

            visited = visited - {nodeId}
            touchingNodes[nodeId] = visited

        return touchingNodes

    def calcModifiedDataLengthForEachnode(self, node):
        if node.isLeaf():
            node.setPruneState(True)
            modifiedDataLength = node.getDatasetSummary().getLength()
            if node.getTouchedNodes() is not None:
                touchedNodes = node.getTouchedNodes()
                for i in touchedNodes:
                    modifiedDataLength += self.cltree.nodes[i].getDatasetSummary().getLength()
            node.setModifiedDataLength(modifiedDataLength)
            return
        else:
            node_L = node.getLeft()
            node_R = node.getRight()
            self.calcModifiedDataLengthForEachnode(node_L)
            self.calcModifiedDataLengthForEachnode(node_R)
            modifiedDataLength = node_L.getModifiedDataLength() + node_R.getModifiedDataLength()
            node.setModifiedDataLength(modifiedDataLength)
            return

    def intializeTouchedNodes(self, root):
        touchingNodes = self.identifyingTouchedRegions(root)
        for nodeId in touchingNodes.keys():
            node = self.cltree.nodes[nodeId]
            touchedNodes = touchingNodes[nodeId]
            for i in touchedNodes:
                node.addTouchedNodes(i)

        self.calcModifiedDataLengthForEachnode(self.root)
        return

    def _accountForTouchedNodesForRepresentationVectors(self, preOrderedListofLeaves, initialRepresentationVectors,
                                                        totalCols):
        representationVectors = {}
        for i in preOrderedListofLeaves:
            node = self.cltree.nodes[i]
            if node.getTouchedNodes() is not None:
                touchedNodes = node.getTouchedNodes()
                touchedNodes.insert(0, i)
                touchedNodes = tuple(sorted(touchedNodes))
                if touchedNodes in representationVectors.keys():
                    continue
                else:
                    instances = np.zeros(len(touchedNodes))
                    c = np.zeros((len(touchedNodes), totalCols))
                    for ii in range(len(touchedNodes)):
                        c_ii = initialRepresentationVectors[touchedNodes[ii]]
                        instances[ii] = self.cltree.nodes[ii].getDatasetSummary().getLength()
                        c[ii, ] = c_ii
                    totalInstances = np.sum(instances)
                    normalizedInstances = instances / totalInstances
                    normalizedInstances = normalizedInstances.reshape((len(touchedNodes), 1)) *\
                                                            np.ones((len(touchedNodes), totalCols))
                    c = c / normalizedInstances
                    vector = np.sum(c, axis=0)
                    representationVectors[touchedNodes] = vector
            else:
                key = tuple([i])
                representationVectors[key] = np.array(initialRepresentationVectors[i])
        return representationVectors

    def _calcInitialRepresentationVectors(self, preOrderedListofLeaves, idColPos):
        eps = 0.0000000001
        conquerData = self.conquerData.copy(deep=True)
        for col in conquerData.columns:
            conquerData[col] = (conquerData[col]-min(self.conquerData[col]))/\
                                            (max(self.conquerData[col])-min(self.conquerData[col])+eps)
        totalCols = conquerData.shape[1]
        representationVectors = {}
        for i in preOrderedListofLeaves:
            node = self.cltree.nodes[i]
            totalDataPoints = node.getDatasetSummary().getLength()
            assert node.dataset is not None, "No Dataset is attached to the Node with ID: {}".format(node.getId())
            if totalDataPoints > 1:
                ids = list(node.dataset.instance_view[:, idColPos])
            else:
                ids = [node.dataset.instance_view[idColPos]]

            allRelevantData = conquerData[conquerData.index.isin(ids)].copy(deep=True)
            assert len(allRelevantData) == totalDataPoints, \
                            "Some data points which have been used to divide have missing conquer data."

            representationVector = np.sum(allRelevantData.values, axis=0) / totalDataPoints
            representationVectors[i] = representationVector
        
        representationVectors = self._accountForTouchedNodesForRepresentationVectors(
            preOrderedListofLeaves, representationVectors, totalCols)

        return representationVectors

    def initializeRepresentationVectors(self, root):
        assert root.depth == 0, logger.error("Depth of the root of the CLTree should be zero(0)!,\
                either the the 'depth' of each node in CLTree is not assigned properly or the node passed is not root!")
        preOrderedListofLeaves = self.preOrderTraversal(root, [])
        assert self.cltree.dataset is not None, "No Dataset is attached with the CLTree at Tree Level"
        columns = self.cltree.dataset.instance_view.shape[1]  # Highest Column
        allColPos = self.cltree.dataset.attr_idx  # All the attributes used to create the Tree
                                                            # and respective numerical column.
        idColPos = set(range(columns)) - set(allColPos.values())  # Getting the position of Id column.
        assert len(idColPos) == 1, "Something went wrong! 'Id' column is not recognized! Attributes: {}," \
                                   " Total Columns:{}".format(allColPos, columns)
        idColPos = idColPos.pop()  # ID values of each row will be used to retrieve the values
                                   # which will be used to join together leaves to form cluster.
        representationVectors = self._calcInitialRepresentationVectors(preOrderedListofLeaves, idColPos)
        return representationVectors


    def _getUnitVectorOfRepresentationVectors(self, representationVectors):
        eps = 0.000001  # Because of this +ve value,
                        # some values may be different when it was getting calculated vs at final result.
                        # Diff would be very very small, at the degree of approx: 10 ^ -8. But still a diff,
                        # so it is better to clarify. If want to validate, make it 0 and restest.

        assert isinstance(representationVectors, dict), logger.error("This function is expecting a key-value pair!")
        for rv in representationVectors.keys():
            representationVector = representationVectors[rv]
            if not isinstance(representationVector, np.ndarray):
                representationVector = np.array(representationVector)
            magnitude = np.sqrt(np.sum(np.square(representationVector)))
            representationVector = representationVector / (magnitude+eps)
            representationVectors[rv] = representationVector
        return representationVectors

    def _recalculateRepresentationVectors(self, key1, representationVector1, key2, representationVector2):
        dataInstances_1 = 0
        dataInstances_2 = 0
        for i in key1:
            dataInstances_1 += self.cltree.nodes[i].getDatasetSummary().getLength()
        for j in key2:
            dataInstances_2 += self.cltree.nodes[j].getDatasetSummary().getLength()
        totalDataInstances = dataInstances_1 + dataInstances_2
        if not isinstance(representationVector1, np.ndarray) or not isinstance(representationVector2, np.ndarray):
            representationVector1 = np.array(representationVector1)
            representationVector2 = np.array(representationVector2)

        representationVector = representationVector1 * (dataInstances_1 / totalDataInstances) + \
                                            representationVector2 * (dataInstances_2 / totalDataInstances)
        if not isinstance(key1, tuple) or not isinstance(key2, tuple):
            key1 = tuple(key1)
            key2 = tuple(key2)

        key = key1 + key2
        key = tuple(sorted(key))
        assert len(key) == len(set(key)), "ERROR! Some node counted twice at the time of pruning!"
        keyRepresentationVector = {key: representationVector}

        return keyRepresentationVector, totalDataInstances

    def _calcDataInstancesbyEachGroup(self, listOfNodeIds):
        totalDataInstancesbyEachGroup = {}
        for i in listOfNodeIds:
            val = 0
            for ii in i:
                val += self.cltree.nodes[ii].getDatasetSummary().getLength()

            totalDataInstancesbyEachGroup[i] = val

        return totalDataInstancesbyEachGroup

    def pruningTreeByMergingRepresentationVectors(self, originalRepresentationVectors, min_y):
        representationVectors = copy.deepcopy(originalRepresentationVectors)
        originalTotalDataInstancesbyEachGroup = self._calcDataInstancesbyEachGroup(list(representationVectors.keys()))
        totalDataInstancesbyEachGroup = copy.deepcopy(originalTotalDataInstancesbyEachGroup)

        rows = len(representationVectors)
        cols = list(representationVectors.values())[0].shape[0]
        data = np.zeros((rows, cols))
        row = 0
        mapping_keyToPos = {}
        mapping_posToKeys = {}
        vertices = {}
        for key in representationVectors.keys():
            val = representationVectors[key]
            data[row, :] = val
            mapping_keyToPos[key] = row
            mapping_posToKeys[row] = key
            vertices[row] = tuple([row])
            row += 1

        data = (data - data.min(0)) / (data.ptp(0) + 0.00001)  # Here, x.ptp(0) returns the "peak-to-peak"
        # (i.e. the range, max - min) along axis 0. This normalization also guarantees that the minimum value in
        # each column will be 0.
        # Agglomerative clustering on whole data.
        tree = shc.linkage(data, method='ward')
        # dend = shc.dendrogram(tree) # If you want to see the graphical representation, feel free to uncomment it.

        # Now we will visit the tree from bottom-up and retrieve information
        # Need to modify totalDataInstancesbyEachGroup and mapping_posToKeys as we clim-up the tree
        # Once all the groups reach the minimum number of datapoints required, take snapshot and continue further
        # till the point when only 2 groups left.
        # Later we will use this snap-shots to determine what is the ideal number of groups/clusters for the
        # underlying min_y.

        snapshot = {}
        i = 0
        while len(vertices) > 2:
            m = max(vertices.keys())
            vx1 = tree[i, 0]
            vx2 = tree[i, 1]
            newVxKey = m + 1

            vx1Val = vertices.get(int(vx1))
            vx2Val = vertices.get(int(vx2))
            newVxVal = vx1Val + vx2Val

            node1 = mapping_posToKeys.get(int(vx1))
            node2 = mapping_posToKeys.get(int(vx2))
            newNode = tuple(sorted(node1 + node2))

            pop1 = totalDataInstancesbyEachGroup.get(node1)
            pop2 = totalDataInstancesbyEachGroup.get(node2)
            newPopulation = pop1+pop2


            del vertices[vx1]
            del vertices[vx2]
            vertices[newVxKey] = newVxVal

            del mapping_posToKeys[vx1]
            del mapping_posToKeys[vx2]
            mapping_posToKeys[newVxKey] = newNode

            del totalDataInstancesbyEachGroup[node1]
            del totalDataInstancesbyEachGroup[node2]
            totalDataInstancesbyEachGroup[newNode] = newPopulation

            if min(totalDataInstancesbyEachGroup.values()) >= min_y:
                snapshot[len(vertices)] = copy.deepcopy(totalDataInstancesbyEachGroup)

            else:
                lengths = list(totalDataInstancesbyEachGroup.values())
                lengths = [l for l in lengths if l < min_y]
                if sum(lengths) < min_y:
                    notQualifiedGroups = {k: v for k, v in totalDataInstancesbyEachGroup.items() if v < min_y}
                    qualifiedGroups = {k: v for k, v in totalDataInstancesbyEachGroup.items() if v >= min_y}

                    newNode = tuple()
                    length = 0
                    for n in notQualifiedGroups.keys():
                        newNode = newNode + n
                        length = length + notQualifiedGroups[n]
                    newNode = tuple(sorted(newNode))
                    qualifiedGroups[newNode] = length
                    snapshot[len(qualifiedGroups) - 0.01] = copy.deepcopy(qualifiedGroups)

            i += 1


        # VALIDATION FOR NO CLUSTER FOUND
        if max(totalDataInstancesbyEachGroup.values()) < min_y:
            return None, None, None

        # Revisiting the snapshots taken while traversing.
        if not self.useSilhouette:
            bestResult = float('inf')
        else:
            bestResult = float('-inf')

        bestScenario = None
        bestFinalVectors =None

        for key in snapshot.keys():
            scenario = snapshot[key]
            scenario_FinalRepresentationVectors = {}
            tmp_scenario_finalRVToCluster = {}
            clusterNo = 0
            for representationVectorKey_F in scenario.keys():
                tmp_scenario_finalRVToCluster[representationVectorKey_F] = 'Cluster_'+str(clusterNo)
                clusterNo += 1

                l = []
                for representationVectorKey_O in representationVectors.keys():
                    if len(set(representationVectorKey_O) - set(representationVectorKey_F)) == 0:
                        l.append(representationVectorKey_O)
                if len(l) == 0:
                    raise ValueError("The Node keys used to construct the original tree mismatched\
                    from the keys used to group the leaf nodes!")

                elif len(l) == 1:
                    scenario_FinalRepresentationVectors[representationVectorKey_F] = representationVectors[l[0]]
                else:
                    origKey_1 = l[0]
                    origKey_2 = l[1]
                    origRV_1 = representationVectors[origKey_1]
                    origRV_2 = representationVectors[origKey_2]
                    key_RV, _ = self._recalculateRepresentationVectors(origKey_1, origRV_1, origKey_2, origRV_2)
                    for ii in range(2,len(l)):
                        origKey_ii = l[ii]
                        origRV_ii = representationVectors[origKey_ii]
                        key_RV, _ = self._recalculateRepresentationVectors(list(key_RV.keys())[0],
                                                                           list(key_RV.values())[0],
                                                                            origKey_ii, origRV_ii)
                    scenario_FinalRepresentationVectors[representationVectorKey_F] = list(key_RV.values())[0]

            result = self.clusterStatistics(originalRepresentationVectors = copy.deepcopy(representationVectors),
                                            finalRepresentationVectors = copy.deepcopy(scenario_FinalRepresentationVectors),
                                            finalRepresentationVectorsToCluster=copy.deepcopy(tmp_scenario_finalRVToCluster),
                                            totalDataInstancesbyEachGroup = copy.deepcopy(scenario))

            if not self.useSilhouette:
                logger.info("Min_Y: {}, Scenario: {},  #Clusters: {}, Inv-Purity: {}".format(min_y, scenario.values(),\
                                                                                             key, result['purity']))
                # Going more to less number of clusters.
                # If we make 10000 clusters from a 10000 data points the value we are optimizing will be the lowest.
                # Our goal is to make sesible and managable number of clusters where each cluster has atleast certain
                # amount of data points.
                # As this part is part of con

                if result['purity'] <= bestResult + self.gradientTolerance:
                    bestResult = result['purity']
                    bestScenario = scenario
                    bestFinalVectors = scenario_FinalRepresentationVectors
            else:
                logger.info("Min_Y: {}, Scenario: {},  #Clusters: {}, silhouette-indicator: {}".format(min_y,
                                                                        scenario.values(), key, result['silhouette']))
                if result['silhouette'] >= bestResult:
                    bestResult = result['silhouette']
                    bestScenario = scenario
                    bestFinalVectors = scenario_FinalRepresentationVectors


        finalRepresentationVectorsToCluster = self._assigingClusterToInternalNodes(bestScenario)

        return bestScenario, bestFinalVectors, finalRepresentationVectorsToCluster

    def _assigingClusterToInternalNodes(self, scenario):
        repVecKeyToCluster = {}
        for repVecKey in scenario.keys():
            if scenario[repVecKey] >= self.min_y:
                clusterID = self.getClusterID()
            else:
                clusterID = self.prefixString + '_' + "cluster_" + "DEFAULT"

            repVecKeyToCluster[repVecKey] = clusterID

            for nodeId in repVecKey:
                node = self.cltree.nodes[nodeId]
                node.includedInCluster = True
                node.setPruneState(True)
                node.clusterId = clusterID

        return repVecKeyToCluster

    def preOrderTraversal(self, node, preOrderedList):
        if node.isPrune():
            preOrderedList.append(node.getId())
            return preOrderedList

        elif node.isLeaf():
            if not node.isPrune():
                node.setPruneState(True)
            preOrderedList.append(node.getId())
            return preOrderedList

        else:
            children = node.getChildNodes()
            if len(children) == 1 and not node.isPrune():
                children[0].setPruneState(True)
                node.setPruneState(True)
                preOrderedList.append(node.getId())
                return preOrderedList

            else:
                preOrderedList = self.preOrderTraversal(children[0], preOrderedList)
                preOrderedList = self.preOrderTraversal(children[1], preOrderedList)

                return preOrderedList

    def preOrderTraversalofClusters(self, node, preOrderedListofClusters):
        if node.includedInCluster:
            preOrderedListofClusters.append((node.parent.attribute, node.clusterId, node))
            return preOrderedListofClusters
        else:
            children = node.getChildNodes()
            if len(children) == 0:
                return preOrderedListofClusters

            preOrderedListofClusters = self.preOrderTraversalofClusters(children[0], preOrderedListofClusters)
            if len(children) == 2:
                preOrderedListofClusters = self.preOrderTraversalofClusters(children[1], preOrderedListofClusters)
            return preOrderedListofClusters

    def leafPruning(self, node):
        if node.includedInCluster:
            return True
        else:
            children = node.getChildNodes()
            lc = self.leafPruning(children[0])
            rc = self.leafPruning(children[1])
            if lc and rc:
                lcId = children[0].clusterId
                rcId = children[1].clusterId
                if lcId == rcId:
                    node.includedInCluster = True
                    node.clusterId = rcId
                    node.setPruneState(True)
                    try:
                        node.dataset = readPicklefile(node.getDatasetPhysicalAddress())
                    except FileNotFoundError:
                        node.dataset = self._merging2Datasets(children[0].dataset, children[1].dataset)
                    return True
                else:
                    return False
            else:
                return False

    def _whichChild(self, node):
        flag = None
        parent = node.parent
        children = parent.getChildNodes()
        if children[0] is node:
            flag = 1
        elif children[1] is node:
            flag = 2
        return flag

    def resetAllPruneNodes(self, node):
        if node.isPrune() or node.includedInCluster:
            node.setPruneState(False)
            node.includedInCluster = False
            node.clusterId = None
            node.touchedNodes = None
            node.modifiedLength = None

        children = node.getChildNodes()
        if len(children) == 0:
            return

        elif len(children) == 1:
            self.resetAllPruneNodes(children[0])

        else:
            self.resetAllPruneNodes(children[0])
            self.resetAllPruneNodes(children[1])

    def clusterStatistics(self, originalRepresentationVectors, finalRepresentationVectors,
                          finalRepresentationVectorsToCluster, totalDataInstancesbyEachGroup):

        originalRepresentationVectors = self._getUnitVectorOfRepresentationVectors(originalRepresentationVectors)
        finalRepresentationVectors = self._getUnitVectorOfRepresentationVectors(finalRepresentationVectors)
        result = {}
        totalDataPointsByClusters = {}
        finalToOriginalRepresentationVectorsMap = {}

        for representationVectorKey_F in finalRepresentationVectors.keys():
            l = []
            for representationVectorKey_O in originalRepresentationVectors.keys():
                if len(set(representationVectorKey_O) - set(representationVectorKey_F)) == 0:
                    l.append(representationVectorKey_O)
            finalToOriginalRepresentationVectorsMap[representationVectorKey_F] = l
            cluster = finalRepresentationVectorsToCluster[representationVectorKey_F]
            totalDataPoints = totalDataInstancesbyEachGroup[representationVectorKey_F]
            totalDataPointsByClusters[cluster] = totalDataPoints
        if not self.useSilhouette:
            allCombosOfFinalRepresentationVectors =\
                sum([list(map(list, combinations(list(finalRepresentationVectors.keys()), i))) for i in range(2, 3)], [])
            # interClusterDistance = Avg. distance between clusters. More the better.
            # Because you want to maintain as much heterogeneity among different clusters.
            # intraClusterDistance = Avg. distance within clusters. Less the better.
            # Because you want maintain as much homogeneity as possible in a cluster.
            interClusterDistance, intraClusterDistance = self._calcClusterStatistics(allCombosOfFinalRepresentationVectors,
                                                                                    finalRepresentationVectors,
                                                                                    originalRepresentationVectors,
                                                                                    finalToOriginalRepresentationVectorsMap)
            # purity = Actually inverse purity, Less the better. Formula: intraClusterDistance/interClusterDistance
            purity = intraClusterDistance / interClusterDistance
            result = {'intra-cluster-distance': intraClusterDistance,
                      'inter-cluster-distance': interClusterDistance,
                      'purity': purity,
                      'data-points': totalDataPointsByClusters}
        else:
            individualPairWiseFinalRepresentationVectors = {}
            finalRepresentationVectorsKeys = list(finalRepresentationVectors.keys())
            for i in finalRepresentationVectorsKeys:
                pairs = []
                finalRepresentationVectorsKeysCopy = finalRepresentationVectorsKeys.copy()
                finalRepresentationVectorsKeysCopy.remove(i)
                pairs = [[i, j] for j in finalRepresentationVectorsKeysCopy]
                individualPairWiseFinalRepresentationVectors[i] = pairs

            silhouetteConstants = self._calcSilhouetteConstants(
                                                            individualPairWiseFinalRepresentationVectors,
                                                            finalRepresentationVectors, originalRepresentationVectors,
                                                            finalToOriginalRepresentationVectorsMap)

            silhouetteIndicator = np.mean(list(silhouetteConstants.values()))

            result = {'silhouetteConstants': silhouetteConstants,
                      'silhouette': silhouetteIndicator,
                      'data-points': totalDataPointsByClusters}

        return result

    def _calcSilhouetteConstants(self, individualPairWiseFinalRepresentationVectors, finalRepresentationVectors,
                               originalRepresentationVectors, finalToOriginalRepresentationVectorsMap):

        individualIntraClusterDistance = {}
        for finalCluster in finalToOriginalRepresentationVectorsMap.keys():
            finalClusterRepresentationVectors = finalRepresentationVectors[finalCluster]
            dist = 0.
            for origGroup in finalToOriginalRepresentationVectorsMap[finalCluster]:
                origClusterRepresentationVectors = originalRepresentationVectors[origGroup]
                dist += self._calcEuclidianDistance(finalClusterRepresentationVectors, origClusterRepresentationVectors)

            dist = dist / len(finalToOriginalRepresentationVectorsMap[finalCluster])
            individualIntraClusterDistance[finalCluster] = dist

        individualInterClusterDistance = {}
        for finalCluster in individualPairWiseFinalRepresentationVectors.keys():
            interClusterDistance = 0.
            pairs = individualPairWiseFinalRepresentationVectors[finalCluster]
            for pair in pairs:
                fc1 = finalRepresentationVectors[pair[0]]
                fc2 = finalRepresentationVectors[pair[1]]
                dist = self._calcEuclidianDistance(fc1, fc2)
                interClusterDistance += dist

            interClusterDistance = interClusterDistance/len(pairs)
            individualInterClusterDistance[finalCluster] = interClusterDistance

        silhouetteConstants = {}
        for finalCluster in finalRepresentationVectors.keys():
            silhouetteConstant = \
                (individualInterClusterDistance[finalCluster] - individualIntraClusterDistance[finalCluster])/\
                np.maximum(individualInterClusterDistance[finalCluster], individualIntraClusterDistance[finalCluster])

            silhouetteConstants[finalCluster] = silhouetteConstant

        return silhouetteConstants

    def _calcClusterStatistics(self, allCombosOfFinalRepresentationVectors, finalRepresentationVectors,
                               originalRepresentationVectors, finalToOriginalRepresentationVectorsMap):

        interClusterDistance = 0.  # Avg. distance between clusters. More the better.
        # Because you want to maintain as much heterogeneity among different clusters.
        intraClusterDistance = 0.  # Avg. distance within clusters. Less the better.
        # Because you want maintain as much homogeneity as possible in a cluster.
        for combos in allCombosOfFinalRepresentationVectors:
            fc1 = finalRepresentationVectors[combos[0]]
            fc2 = finalRepresentationVectors[combos[1]]
            dist = self._calcEuclidianDistance(fc1, fc2)
            interClusterDistance += dist
        interClusterDistance = interClusterDistance / len(finalRepresentationVectors.keys())

        individualIntraClusterDistance = {}
        for finalCluster in finalToOriginalRepresentationVectorsMap.keys():
            finalClusterRepresentationVectors = finalRepresentationVectors[finalCluster]
            dist = 0.
            for origGroup in finalToOriginalRepresentationVectorsMap[finalCluster]:
                origClusterRepresentationVectors = originalRepresentationVectors[origGroup]
                dist += self._calcEuclidianDistance(finalClusterRepresentationVectors, origClusterRepresentationVectors)

            dist = dist / len(finalToOriginalRepresentationVectorsMap[finalCluster])
            individualIntraClusterDistance[finalCluster] = dist

        intraClusterDistance = sum(individualIntraClusterDistance.values()) / len(finalRepresentationVectors.keys())
        return interClusterDistance, intraClusterDistance

    def _calcEuclidianDistance(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def getClusterID(self):
        self.maxSoFar += 1
        return self.prefixString + '_' + "cluster_" + str(self.maxSoFar)

    def pruningRedundantNodes(self, root):
        preOrderedListofClusters = self.preOrderTraversalofClusters(root, [])
        preOrderedListofClusters = [node for _, _, node in preOrderedListofClusters]
        if len(preOrderedListofClusters) < 3:
            return

        curPosition = 0
        nextPosition = 1
        while True:
            if nextPosition <= curPosition or nextPosition >= len(preOrderedListofClusters):
                break

            curNode = preOrderedListofClusters[curPosition]
            nxtNode = preOrderedListofClusters[nextPosition]
            curNodeParent = curNode.getParent()
            nxtNodeParent = nxtNode.getParent()
            curNodeWhichChild = self._whichChild(curNode)
            nxtNodeWhichChild = self._whichChild(nxtNode)
            if curNodeWhichChild != nxtNodeWhichChild or curNode.clusterId != nxtNode.clusterId or \
                    curNode.parent.attribute != nxtNode.parent.attribute:
                curPosition += 1
                nextPosition += 1
                continue
            else:
                if curNodeWhichChild == 1:
                    # Both are left child of their respective parents.
                    nxtNodeGrandParent = nxtNode.getParent().getParent()
                    if curNodeParent == nxtNodeGrandParent and \
                            curNodeParent is not None and nxtNodeGrandParent is not None:
                        # Replacing curNodeParent and curNode by
                        # curNodeParent_RightChild(sibling of curNode) and
                        # curNodeParent_RightChild_LeftChild(nxtNode) respectively.
                        curNodeGrandParent = curNodeParent.getParent()
                        if curNodeGrandParent is not None:
                            curNodeParentWhichChild = self._whichChild(curNodeParent)
                            curNodeGrandParent.children[curNodeParentWhichChild - 1] = nxtNodeParent
                        else:
                            self.root = nxtNodeParent
                            self.cltree.root = nxtNodeParent
                            allNodes = self.cltree._collectAllNodes(self.cltree.root, dict())
                            self.cltree.nodes = allNodes

                        nxtNodeParent.parent = curNodeGrandParent
                        datasetId = curNodeParent.getDatasetId()
                        datasetPhysicalAddress = curNodeParent.getDatasetPhysicalAddress()
                        datasetSummary = curNodeParent.getDatasetSummary()
                        nxtNodeParent.modifyUnderlyingDataSet(datasetId, datasetPhysicalAddress)
                        nxtNodeParent.datasetSummary = datasetSummary

                        nxtNodeDataSet = self._merging2Datasets(curNode.dataset, nxtNode.dataset)
                        nxtNodeDataSet.calculate_limits()
                        attr_names = nxtNodeDataSet.attr_names
                        attr_idx = nxtNodeDataSet.attr_idx
                        max_values = nxtNodeDataSet.max_values
                        min_values = nxtNodeDataSet.min_values
                        length = nxtNodeDataSet.getLength()
                        nxtNodeDataSetId = str(curNode.getDatasetSummary().getId()) + "_AND_" + \
                                           str(nxtNode.getDatasetSummary().getId())
                        nxtNodeDataSetPhysicalAddress = self.tmpFilesPath + nxtNodeDataSetId + ".pickle"
                        writePickleFile(path=nxtNodeDataSetPhysicalAddress, data=nxtNodeDataSet)
                        nxtNodeDataSetSummay = DataSetSummary(nxtNodeDataSetId, nxtNodeDataSetPhysicalAddress,
                                                              attr_names, attr_idx, max_values, min_values, length)
                        nxtNode.modifyUnderlyingDataSet(nxtNodeDataSetId, nxtNodeDataSetPhysicalAddress)
                        nxtNode.datasetSummary = nxtNodeDataSetSummay

                        preOrderedListofClusters.remove(curNode)

                    else:
                        curPosition += 1
                        nextPosition += 1

                elif curNodeWhichChild == 2:
                    # Both are right child of their respective parents.
                    curNodeGrandParent = curNodeParent.getParent()
                    if curNodeGrandParent == nxtNodeParent and \
                            nxtNodeParent is not None and curNodeGrandParent is not None:
                        # Replacing curNodeGrandParent/nxtNodeParent by curParent
                        nxtNodeGrandParent = nxtNodeParent.getParent()
                        if nxtNodeGrandParent is not None:
                            nxtNodeParentWhichChild = self._whichChild(nxtNodeParent)
                            nxtNodeGrandParent.children[nxtNodeParentWhichChild - 1] = curNodeParent
                        else:
                            self.root = curNodeParent
                            self.cltree.root = curNodeParent
                            allNodes = self.cltree._collectAllNodes(self.cltree.root, dict())
                            self.cltree.nodes = allNodes

                        curNodeParent.parent = nxtNodeGrandParent
                        datasetId = nxtNodeParent.getDatasetId()
                        datasetPhysicalAddress = nxtNodeParent.getDatasetPhysicalAddress()
                        datasetSummary = nxtNodeParent.getDatasetSummary()
                        curNodeParent.modifyUnderlyingDataSet(datasetId, datasetPhysicalAddress)
                        curNodeParent.datasetSummary = datasetSummary

                        curNodeDataSet = self._merging2Datasets(curNode.dataset, nxtNode.dataset)
                        curNodeDataSet.calculate_limits()
                        attr_names = curNodeDataSet.attr_names
                        attr_idx = curNodeDataSet.attr_idx
                        max_values = curNodeDataSet.max_values
                        min_values = curNodeDataSet.min_values
                        length = curNodeDataSet.getLength()
                        curNodeDataSetId = str(curNode.getDatasetSummary().getId()) + "_AND_" + \
                                           str(nxtNode.getDatasetSummary().getId())
                        curNodeDataSetPhysicalAddress = self.tmpFilesPath + curNodeDataSetId + ".pickle"
                        writePickleFile(path=curNodeDataSetPhysicalAddress, data=curNodeDataSet)
                        curNodeDataSetSummay = DataSetSummary(curNodeDataSetId, curNodeDataSetPhysicalAddress,
                                                              attr_names, attr_idx, max_values, min_values, length)
                        curNode.modifyUnderlyingDataSet(curNodeDataSetId, curNodeDataSetPhysicalAddress)
                        curNode.datasetSummary = curNodeDataSetSummay

                        preOrderedListofClusters.remove(nxtNode)

                    else:
                        curPosition += 1
                        nextPosition += 1

        self._recalculateDepthOfEachNodes(self.cltree.root, 0)
        return

    def _recalculateDepthOfEachNodes(self, node, depth):
        node.depth = depth
        childrens = node.getChildNodes()
        if childrens is not None and len(childrens) > 0:
            self._recalculateDepthOfEachNodes(childrens[0], depth + 1)
        if childrens is not None and len(childrens) > 1:
            self._recalculateDepthOfEachNodes(childrens[1], depth + 1)
        return

    def _merging2Datasets(self, dataset1, dataset2):
        assert isinstance(dataset1, Data) and isinstance(dataset2, Data), \
            "_merging2Datasets() can handle only instance of ADT 'Data' class!"
        dataset1_instance_values = dataset1.instance_values
        dataset1_attr_types = dataset1.attr_types
        dataset1_attr_idx = dataset1.attr_idx

        dataset2_instance_values = dataset2.instance_values
        dataset2_attr_types = dataset2.attr_types
        dataset2_attr_idx = dataset2.attr_idx

        if len(dataset1_attr_idx) != len(dataset2_attr_idx) and \
                len(set(dataset1_attr_idx.keys()) - set(dataset2_attr_idx.keys())) != 0:
            logger.error('Could not merge 2 datasets due to datatype/column mismatch.\
                                    dataset1 attr. types: {}, dataset2 attr. types: {}'.format(dataset1_attr_types,
                                                                                               dataset2_attr_types))
            raise ValueError('Could not merge 2 datasets due to datatype/column mismatch.\
                                    dataset1 attr. types: {}, dataset2 attr. types: {}'.format(dataset1_attr_types,
                                                                                               dataset2_attr_types))
        else:
            for k in dataset1_attr_idx.keys():
                dataset1_k_idx = dataset1_attr_idx[k]
                dataset2_k_idx = dataset2_attr_idx[k]
                if dataset1_k_idx != dataset2_k_idx:
                    temp = np.copy(dataset2_instance_values[:, dataset1_k_idx])
                    dataset2_instance_values[:, dataset1_k_idx] = dataset2_instance_values[:, dataset2_k_idx]
                    dataset2_instance_values[:, dataset2_k_idx] = temp

            newDataSetInstances = np.concatenate((dataset1_instance_values, dataset2_instance_values), axis=0)
            dataset = Data(newDataSetInstances, dataset1_attr_types)
            return dataset


def pruneByGridSearch(cltree, min_y, prefixString, gradientTolerance, conquerData, divideData, useSilhouette):
    assert isinstance(cltree, CLTree) and isinstance(cltree.getRoot(), CLNode), logger.error(\
               "ERROR! Data Type mismatch. This function can operate only on CLTree and CLNode")
    root = cltree.getRoot()
    minLowerLimit = min_y - int(min_y * 0.1)
    p = PruneTreeByConqueringNodes(prefixString, cltree, conquerData.copy(deep=True), useSilhouette)
    p.resetAllPruneNodes(node=root)
    result = p.prune(min_y=minLowerLimit, gradientTolerance=gradientTolerance, searchMode=False)
    if result is None:
        logger.warning("No meaningful clusters found by 'conquerData'. Falling back to 'divideData'\ "
                       "to find meaningful clusters with atleast {} data points.".format(minLowerLimit))
        p = PruneTreeByConqueringNodes(prefixString, cltree, divideData, useSilhouette)
        p.resetAllPruneNodes(node=root)
        result = p.prune(min_y=minLowerLimit, gradientTolerance=0., searchMode=False)

        # Result will be None, if algorithm couldn't find meaningful clusters each containing at-least minLowerLimit.
        if result is None:
            logger.error("No meaningful clusters found either by 'conquerData' or by 'divideData'\
                                with atleast {} data points. Recommended action, \
                                           try 'Balanced' clustering the same data.".format(minLowerLimit))
            raise RuntimeError("No meaningful clusters found either by 'conquerData' or by 'divideData' \
                               with atleast {} data points. Recommended action, \
                               try 'Balanced' clustering the same data.".format(minLowerLimit))

    cltree.setClustersStatistics(result)
    baseVer = p.version
    return result, baseVer

