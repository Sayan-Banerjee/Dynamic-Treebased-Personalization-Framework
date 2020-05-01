import logging
import pandas as pd
from math import sqrt as sqrt
import numpy as np
import copy

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _relative_density(dataset):
    return float(dataset.length())/dataset.nr_virtual_points

class Data:
    '''DataFrame Data'''
    def __init__(self, instance_values, types):
        self.instance_values = instance_values
        self.attr_types = types
        self.attr_idx = dict()
        self.attr_names = list()

        self._init_attr_names()
        self._init_max_min() # Could replace with self.calculate_limits()

        self.nr_virtual_points = len(self.instance_values)
        self.nr_total_instances = 2*self.nr_virtual_points

    def _init_max_min(self):
        if len(self.instance_values) > 1:
            self.instance_view = self.instance_values.view(dtype=float).reshape(len(self.instance_values),-1)
            self.max_values = np.amax(self.instance_view, 0)
            self.min_values = np.amin(self.instance_view, 0)
        else:
            self.instance_view = self.instance_values.view(dtype=float)
            self.max_values = copy.copy(self.instance_view)
            self.min_values = copy.copy(self.instance_view)

    def _init_attr_names(self):
        for i, attr in enumerate(self.attr_types):
            if i < 1:
                continue
            attr_name, attr_type = attr
            self.attr_idx[attr_name] = i
            self.attr_names.append(attr_name)

    def __str__(self):
        s = 'Data: ' + str(len(self.instance_values)) + "\n"
        s += str(self.attr_names) + "\n"
        s += " Max :" + str(self.max_values)+ "\n"
        s += " Min :" + str(self.min_values)+ "\n"
        # Commenting it out for now as I am using this in logger to convey the information at any the bottom-most state of a tree.
        #s += str(self.instance_values)
        s += '\n--------\n'
        return s

    def calculate_limits(self):
        self._init_max_min()

    def sort(self, attribute):
        self.instance_values = np.sort(self.instance_values, order=attribute)
        self.instance_view = self.instance_values.view(dtype=float).reshape(len(self.instance_values), -1)

    def length(self):
        return len(self.instance_values)

    def getInstances(self, attribute):
        idx = self.attr_idx[attribute]
        if self.length() > 1:
            return self.instance_view[:,idx]
        elif self.length() == 1:
            return [self.instance_view[0,idx]]
        else:
            return []

    def getInstanceIndex(self, id):
        if self.length() > 1:
            idx = np.argwhere(self.instance_view[:,0] == id)
            return idx[0]
        elif self.length() == 1 and id == self.instance_view[0]:
            return 0
        else:
            return None

    def getId(self, idx):
        if self.length() > 1:
            return self.instance_view[idx][0]
        elif self.length() == 1:
            return self.instance_view[0]
        else:
            return -1

    def get_max(self, attribute):
        idx = self.attr_idx[attribute]
        return self.max_values[idx]

    def get_min(self, attribute):
        idx = self.attr_idx[attribute]
        return self.min_values[idx]

    def set_max(self, attribute, value):
        if len(self.max_values) > 0:
            idx = self.attr_idx[attribute]
            self.max_values[idx] = value

    def set_min(self, attribute, value):
        if len(self.min_values) > 0:
            idx = self.attr_idx[attribute]
            self.min_values[idx] = value

class DataFrameReader:
    def __init__(self, df):
        self.df = df
        pass

    def read(self):
        types = self._read_columnNames()
        output = self._read_instances()

        output = np.array(output, dtype=types)
        data = Data(output, types)

        logger.info("read " + str(len(self.df)))
        logger.info("attribute names: {}".format(data.attr_types))

        return data

    def _read_columnNames(self):
        dtype = list()
        dtype.append(("id", float))
        for col in list(self.df.columns):
            dtype.append((col, float))

        return dtype

    def _read_instances(self):
        output = list()
        count = 0.0
        for index, row in self.df.iterrows():
            a = tuple(row.values)
            a = tuple([count]) + a
            output.append(a)
            count += 1.0
        return output

class DatasetSplitter:
    def __init__(self):
        pass

    def split(self, dataset, attribute, value, idx):
        l = dataset.instance_values[0: idx]

        r = dataset.instance_values[idx:]
        lhs_set = Data(l, dataset.attr_types)
        rhs_set = Data(r, dataset.attr_types)

        # Set the Lhs min and Max and Rhs Max.
        lhs_set.calculate_limits()
        rhs_set.calculate_limits()

        self._splitNrVirtualPoints(dataset, attribute, value, lhs_set, rhs_set)

        self._updateVirtualPoints(lhs_set)
        self._updateVirtualPoints(rhs_set)

        return lhs_set, rhs_set

    def _splitNrVirtualPoints(self, dataset, attribute, value, in_set, out_set):
        minV = dataset.get_min(attribute)
        maxV = dataset.get_max(attribute)

        in_set.nr_virtual_points = int(abs(dataset.nr_virtual_points*((value-minV)/(maxV-minV))))
        out_set.nr_virtual_points = dataset.nr_virtual_points - in_set.nr_virtual_points
        if out_set.nr_virtual_points < 0:
            self.raiseUndefinedNumberOfPoints()

    def _updateVirtualPoints(self, data_set):
        nr_points_in_set = data_set.length()
        data_set.nr_virtual_points = self._calcNumberOfPointsToAdd(nr_points_in_set, data_set.nr_virtual_points)
        data_set.nr_total_instances = nr_points_in_set + data_set.nr_virtual_points

    def _calcNumberOfPointsToAdd(self, nr_points_in_node, nr_points_inherited):
        if nr_points_inherited < nr_points_in_node:
            nr_points = nr_points_in_node
        else:
            nr_points = nr_points_inherited
        return nr_points

    def raiseUndefinedNumberOfPoints(self):
        raise DatasetSplitter.UndefinedNumberOfPoints()
    class UndefinedNumberOfPoints(Exception):
        pass

class BuildTree(object):
    def __init__(self, min_split):
        self.cutCreator = InfoGainCutFactory(min_split)
        self.datasetSplitter = DatasetSplitter()
        self.min_split = min_split
        self.root = None
        self.maxNodeId = 0

    def build(self, dataset):
        self._build_tree(dataset, None, 0)
        return self.root

    def _build_tree(self, dataset, parent, depth):
        if parent is not None:
            logger.info("At previous level({}), the cut happened on Attribute:{}, Value: {}".
                   format(parent.depth, parent.attribute, parent.cutValue))
        else:
            logger.info("This is the starting point!")

        bestCut = self._findBestCut(dataset)
        cutValue = None if bestCut is None else bestCut.value
        attribute = "" if bestCut is None else bestCut.attribute

        if bestCut is not None:
            logger.info("At this level the best cut is found on Attribute: {}, at: {}".format(attribute, cutValue))
        else:
            logger.info("No cut is found at current level!")

        dt_node = CLNode(dataset, parent, attribute, depth, cutValue)
        dt_node.setID(self.maxNodeId)
        self.maxNodeId+=1

        if parent: parent.addChildNode(dt_node)

        if self._isRootNode(depth): self.root = dt_node

        if bestCut is None:
            return

        lhs_dataset, rhs_dataset = self._splitDatasetUsingBestCut(dataset, bestCut)

        if lhs_dataset.length() > self.min_split:
            self._build_tree(lhs_dataset, dt_node, (depth+1))
        else:
            logger.info ("Inadequate data points to create further split on left child!")
            logger.info ("Level: {}, Datapoints: \n{}".format(depth+1, lhs_dataset.__str__()))

            lcNode = CLNode(lhs_dataset, dt_node, None, depth+1, None)
            dt_node.addChildNode(lcNode)
            lcNode.setID(self.maxNodeId)
            self.maxNodeId+=1
        if rhs_dataset.length() > self.min_split:
            self._build_tree(rhs_dataset, dt_node, (depth+1))
        else:
            logger.info ("Inadequate data points to create further split on right child!")
            logger.info ("Level: {}, Datapoints: \n{}".format(depth+1, rhs_dataset.__str__()))

            rcNode = CLNode(rhs_dataset, dt_node, None, depth+1, None)
            dt_node.addChildNode(rcNode)
            rcNode.setID(self.maxNodeId)
            self.maxNodeId+=1
        return

    def _isRootNode(self, depth):
        if depth==0 and self.root is None: return True

    def _splitDatasetUsingBestCut(self, dataset, bestCut):
        dataset.sort(bestCut.attribute)
        idx = dataset.getInstanceIndex(bestCut.inst_id)
        try:
            idx = idx[-1]
        except:
            pass

        lhs_set, rhs_set = self.datasetSplitter.split(dataset, bestCut.attribute, bestCut.value, idx+1)
        lhs_set.calculate_limits()
        rhs_set.calculate_limits()

        return lhs_set, rhs_set

    def _findBestCut(self, dataset):
        dataset.calculate_limits()
        bestCut = None
        for attribute in dataset.attr_names:
            dataset.sort(attribute) # Sorting the dataset based on a attribute value at a time.
            di_cut1 = self._calcCut1(dataset, attribute) # Best cut on current attribute based on "Information Gain"
            if di_cut1 is None: # Ignore dimension go to the next dimension.
                continue
            di_cut2 = self._calcCut2(di_cut1) # Acquiring a second cut on the current attribute at Low Density Region.
            if di_cut2 is None: # If there is no di_cut2 there, di_cut1 is the best cut on this dimension.
                bestCut = self._selectLowerDensityCut(di_cut1, bestCut) # Keeping Track of best cut incrementally.
                continue # We are done with this dimension, skip the rest and go to the next dimension.
            di_cut3 = self._calcCut3(di_cut1, di_cut2) # Acquire a cut in the space which is in between di_cut1 and di_cut2.
            if di_cut3 is None: # if di_cut3 is no better than di_cut2 in terms of relative density the di_cut2 is the best cut on this dimesion.
                bestCut = self._selectLowerDensityCut(di_cut2, bestCut) # Keeping Track of best cut incrementally.
            else: # If there is a di_cut3 that means there is a lower density region and di_cut3 is the best cut on this dimension.
                bestCut = self._selectLowerDensityCut(di_cut3, bestCut) # Keeping Track of best cut incrementally.

        return bestCut # Returning the best cut at this level and the local dataset, which is a subset of the entire dataset.

    def _calcCut1(self, dataset, attribute):
        return self.cutCreator.cut(dataset, attribute) # Acquire the first cut on the dimension based on Information Gain

    def _calcCut2(self, di_cut1):
        lower_density_set = di_cut1.getLowerDensityRegion() # Acquire the lower realtive density region among the regions created by di_cut1
        return self.cutCreator.cut(lower_density_set, di_cut1.attribute) # Acquire the second cut on the dimension in the Lower Density Region based on Information Gain.

    def _calcCut3(self, di_cut1, di_cut2):
        adjacentRegion = di_cut2.getAdjacentRegion(di_cut1.value, di_cut1.attribute) # Acquire at which region the value used to acquire the di_cut1 falls in as the result of di_cut2.
        otherRegion = di_cut2.getNonAdjacentRegion(di_cut1.value, di_cut1.attribute) # Acquire at which region the value used to acquire the di_cut1 DOES NOT falls in as the result of di_cut2.
        di_cut3 = None
        if _relative_density(adjacentRegion) <= _relative_density(otherRegion): # Compare relative densities among two regions as the result of di_cut2.
            lower_density_set = di_cut2.getLowerDensityRegion() # Acquire the lower realtive density region among the regions created by di_cut2.
            di_cut3 = self.cutCreator.cut(lower_density_set, di_cut2.attribute) # Acquire the third cut on the dimension in the Lower Density Region based on Information Gain.

        return di_cut3

    def _selectLowerDensityCut(self, cut1, cut2):
        if cut1 is None: return cut2
        if cut2 is None: return cut1
        rd1 = cut1.getRelativeDensityOfLowerDensityRegion() # Acquire the lower density region by cut1
        rd2 = cut2.getRelativeDensityOfLowerDensityRegion() # Acquire the lower density region by cut2
        if rd1 < rd2:
            return cut1 # Return the cut which has the lower density region with the lowest relative density.
        else:
            return cut2

class Cut:
    def __init__(self, attribute, value, inst_id, lhsset, rhsset):
        self.attribute = attribute
        self.value = value
        self.inst_id = inst_id
        self.lhs_set = lhsset
        self.rhs_set = rhsset

    def __str__(self):
        s = 'Cut: ' + self.attribute + "\n"
        s += str(self.lhs_set.attr_names) + "\n"
        s += " Max lhs:" + str(self.lhs_set.max_values)+ "\n"
        s += " Min lhs:" + str(self.lhs_set.min_values)+ "\n"
        s += " Max rhs:" + str(self.rhs_set.max_values)+ "\n"
        s += " Min rhs:" + str(self.rhs_set.min_values)
        s += '\n--------\n'
        return s

    def getNonAdjacentRegion(self, value, attribute):
        dataset = self.getAdjacentRegion(value, attribute)
        if dataset is self.lhs_set:
            return self.rhs_set
        if dataset is self.rhs_set:
            return self.lhs_set
        return None

    def getAdjacentRegion(self, value, attribute):
        def getMinimumDistanceFromValue(dataset, attribute, value):
            distance1 = abs(dataset.get_max(attribute) - value)
            distance2 = abs(dataset.get_min(attribute) - value)
            return min(distance1, distance2)
        rhs_distance = getMinimumDistanceFromValue(self.rhs_set, attribute, value)
        lhs_distance = getMinimumDistanceFromValue(self.lhs_set, attribute, value)

        if lhs_distance < rhs_distance: return self.rhs_set
        else: return self.lhs_set

    def getRelativeDensityOfLowerDensityRegion(self):
        lower_density_set = self.getLowerDensityRegion()
        r_density = _relative_density(lower_density_set)
        return r_density

    def getLowerDensityRegion(self):
        if self.lhs_set is None or self.rhs_set is None:
            self.raiseNoRegionsDefined()

        if _relative_density(self.lhs_set) > _relative_density(self.rhs_set):
            return self.rhs_set
        else:
            return self.lhs_set

    def raiseNoRegionsDefined(self):
        raise Cut.NoRegionsDefined()
    class NoRegionsDefined(Exception):
        pass

class InfoGainCutFactory:
    def __init__(self, min_split):
        self.min_split = min_split
        self.datasetSplitter = DatasetSplitter()

    def cut(self, dataset, attribute):
        dataset.sort(attribute)
        di_cut = None
        max_info_gain = -1
        instances = dataset.getInstances(attribute)
        nunique = np.unique(instances)
        for value in nunique:
            i = np.argwhere(instances == value)
            i = i[-1]
            i = i[0]

            if self._hasRectangle(dataset, attribute, value):
                lhs_set, rhs_set = self.datasetSplitter.split(dataset, attribute, value, i+1)
                ig, lset, rset = self._info_gain(dataset, lhs_set, rhs_set)
                if ig > max_info_gain:
                    max_info_gain = ig
                    di_cut = Cut(attribute, value, dataset.getId(i), lset, rset)

        return di_cut

    def _hasRectangle(self, dataset, attribute, value):
        if dataset.get_max(attribute) == dataset.get_min(attribute):
            return False
        else:
            if dataset.get_max(attribute) == value:
                return False
            else:
                return True

    def _info_gain(self, dataset, lhs_set, rhs_set):
        #if (lhs_set.nr_total_instances < self.min_split) or (rhs_set.nr_total_instances < self.min_split):
        if (lhs_set.nr_total_instances < 1) or (rhs_set.nr_total_instances < 1):
            return -1, lhs_set, rhs_set

        ratio_instances_lhs = (float(lhs_set.nr_total_instances)/dataset.nr_total_instances)
        ratio_instances_rhs = (float(rhs_set.nr_total_instances)/dataset.nr_total_instances)
        entropy2 = ratio_instances_lhs*self._calc_entropy(lhs_set) + ratio_instances_rhs*self._calc_entropy(rhs_set)

        entropy1 = self._calc_entropy(dataset)

        return (entropy1 - entropy2), lhs_set, rhs_set

    def _calc_entropy(self, dataset):
        nr_existing_instances = dataset.length()
        total = nr_existing_instances + dataset.nr_virtual_points
        terms = list()
        # log(base2) is equivalent to sqrt. (e^x)^y = e^xy = (e^y)^x
        terms.append((float(nr_existing_instances)/float(total))*sqrt(float(nr_existing_instances)/float(total)))
        terms.append((float(dataset.nr_virtual_points)/float(total))*sqrt(float(dataset.nr_virtual_points)/float(total)))
        return sum(terms)*-1

class CLNode(object):
    # Added cut to it. So that It can be reused. Cut can be represented by value and attribute.
    def __init__(self, dataset, parent, attribute, depth, cutValue):
        self.dataset = dataset
        self.parent = parent
        self.attribute = attribute
        self.cutValue = cutValue
        self.depth = depth
        self.children = list()
        self.can_prune = False
        self.includedInCluster = False
        self.clusterId = None
        self.nodeId = None
        self.touchedNodes = None
        self.modifiedLength = None

    def setPruneState(self, prune):
        self.can_prune = prune

    def isPrune(self):
        return self.can_prune

    def getRelativeDensity(self):
        return _relative_density(self.dataset)*100.0

    def getNrInstancesInNode(self):
        return self.dataset.length()

    def addChildNode(self, node):
        self.children.append(node)

    def getChildNodes(self):
        return self.children

    def getLeft(self):
        if len(self.children)==0:
            return None
        else:
            return self.children[0]

    def getRight(self):
        if len(self.children)<2:
            return None
        else:
            return self.children[1]

    def getParent(self):
        return self.parent

    def isLeaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False
    def setID(self, nodeId):
        self.nodeId = nodeId
        self.dataset.calculate_limits()

    def getID(self):
        return self.nodeId

    def getTouchedNodes(self):
        return self.touchedNodes

    def addTouchedNodes(self, nodeId):
        if self.touchedNodes is None:
            self.touchedNodes = list()

        self.touchedNodes.append(nodeId)
        return

    def setModifiedDataLength(self, modifiedLength):
        self.modifiedLength = modifiedLength
        return

    def getModifiedDataLength(self):
        return self.modifiedLength

    def raiseAddNode(self):
        raise CLNode.AddNodeIlogical("hi")

    class AddNodeIlogical(Exception):
        pass

    def __str__(self):
        attr = list()
        vals = list()
        p = self.parent
        while p:
            attr.append(p.attribute)
            vals.append(p.cutValue)
            p = p.parent

        s = 'Node: ' + '\n'
        s += str(self.dataset.length()) + ' instances, '
        s += ", " + str(int(self.getRelativeDensity())) + " relative density " + '\n'
        s += ", " + str(self.depth) + " depth " + '\n'
        s += "Cuts " + str(attr)+ " At Values " + str(vals) + '\n'

        self.dataset.calculate_limits()
        try:
            for name in self.dataset.attr_names:
                s += name + ' max: ' + str(self.dataset.get_max(name))+\
                    ' min: ' + str(self.dataset.get_min(name))+'\n'
        except:
            pass

        return s

    def str2(self):
        instance = self.dataset.length()
        s = 'ID: {}'.format(self.nodeId)
        s+= 'Inst: {} '.format(instance)
        self.dataset.calculate_limits()
        try:
            for name in self.dataset.attr_names:
                s += ' ' + name + ' max: ' + str(self.dataset.get_max(name))+\
                    ' min: ' + str(self.dataset.get_min(name))+ '\n'
        except:
            pass
        s+= '' if self.clusterId is None else self.clusterId
        return s

class CLTree:
    def __init__(self, dataset, min_split=100):
        if dataset is None:
            self.raiseUndefinedDataset()
        self.dataset = dataset
        self.min_split = min_split
        self.clusters = dict()
        self.nodes = dict()
        self.clustersStatistics = None
        self.root = None

    def buildTree(self):
        b = BuildTree(self.min_split)
        self.root = b.build(self.dataset)
        allNodes = self._collectAllNodes(self.root, dict())
        self.nodes  = allNodes

    def getRoot(self):
        return self.root

    def getAllClusters(self):
        return self.clusters

    def getAllNodes(self):
        return self.nodes

    def getClustersStatistics(self):
        return self.clustersStatistics

    def setClustersStatistics(self, result):
        self.clustersStatistics = result
        allClusters = self._collectAllClusters(self.root, dict())
        self.clusters = allClusters
        return

    def _collectAllClusters(self, node, allClusters):

        if node.includedInCluster:
            clusterId = node.clusterId
            if clusterId in allClusters.keys():
                val = allClusters[clusterId]
                val.append(node.nodeId)
                allClusters[clusterId] = val
            else:
                allClusters[clusterId] = [node.nodeId]

        else:
            children = node.getChildNodes()
            self._collectAllClusters(children[0], allClusters)
            self._collectAllClusters(children[1], allClusters)

        return allClusters

    def _collectAllNodes(self, node, allNodes):
        # In-order Traversal
        if node.isLeaf():
            allNodes[node.nodeId] = node
            return allNodes

        children = node.getChildNodes()
        allNodes = self._collectAllNodes(children[0], allNodes)
        allNodes[node.nodeId] = node
        allNodes = self._collectAllNodes(children[1], allNodes)
        return allNodes

    def raiseUndefinedDataset(self):
        raise CLTree.UndefinedDataset()

    class UndefinedDataset(Exception):
        pass

    def raiseUndefinedTree(self):
        raise CLTree.UndefinedTree()

    class UndefinedTree(Exception):
        pass
