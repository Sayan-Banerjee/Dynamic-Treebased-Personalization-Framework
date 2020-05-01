import logging
import numpy as np
import copy

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

from ReadWritePickleFile import readPicklefile, writePickleFile
from CLTree_Utility import doThisPathExist, createPath, getRandomStringasId

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _relative_density(dataset):
    return float(dataset.getLength()) / dataset.nr_virtual_points


class Data:
    '''DataFrame Data'''

    def __init__(self, instance_values, types):

        self.instance_values = instance_values
        self.attr_types = types
        self.attr_idx = dict()
        self.attr_names = list()
        self.totalNumberofDataPoints = None
        self._init_attr_names()
        self._init_max_min()  # Could replace with self.calculate_limits()
        self.nr_virtual_points = len(self.instance_values)
        self.nr_total_instances = 2 * self.nr_virtual_points

    def _init_max_min(self):
        if len(self.instance_values) > 1:
            self.instance_view = self.instance_values.view(dtype=float).reshape(len(self.instance_values), -1)
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
        return

    def __str__(self):
        s = 'Data: ' + str(len(self.instance_values)) + "\n"
        s += str(self.attr_names) + "\n"
        s += " Max :" + str(self.max_values) + "\n"
        s += " Min :" + str(self.min_values) + "\n"
        s += '\n--------\n'
        return s

    def calculate_limits(self):
        self._init_max_min()

    def sort(self, attribute):
        self.instance_values = np.sort(self.instance_values, order=attribute)
        self.instance_view = self.instance_values.view(dtype=float).reshape(len(self.instance_values), -1)

    def getLength(self):
        if self.totalNumberofDataPoints is None:
            self.totalNumberofDataPoints = len(self.instance_values)

        return self.totalNumberofDataPoints

    def getInstances(self, attribute):
        idx = self.attr_idx[attribute]
        if self.getLength() > 1:
            return self.instance_view[:, idx]
        elif self.getLength() == 1:
            return [self.instance_view[0, idx]]
        else:
            return []

    def getInstanceIndex(self, id):
        if self.getLength() > 1:
            idx = np.argwhere(self.instance_view[:, 0] == id)
            return idx[0]
        elif self.getLength() == 1 and id == self.instance_view[0]:
            return 0
        else:
            return None

    def getId(self, idx):
        if self.getLength() > 1:
            return self.instance_view[idx][0]
        elif self.getLength() == 1:
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


class DataSetSummary:

    def __init__(self, dataSetId, datasetPhysicalAddress, attr_names, attr_idx, max_values, min_values, length):
        self.dataSetId = dataSetId
        self.datasetPhysicalAddress = datasetPhysicalAddress
        self.attr_names = attr_names
        self.attr_idx = attr_idx
        self.max_values = max_values
        self.min_values = min_values
        self.length = length
        return

    def get_max(self, attribute):
        idx = self.attr_idx[attribute]
        return self.max_values[idx]

    def get_min(self, attribute):
        idx = self.attr_idx[attribute]
        return self.min_values[idx]

    def getLength(self):
        return self.length

    def getId(self):
        return self.dataSetId

    def getPhysicalAddress(self):
        return self.datasetPhysicalAddress

    def getAttributes(self):
        return self.attr_names

    def getAttrIndexes(self):
        return self.attr_idx

    def __str__(self):
        s = "Data Set Id: {} ".format(self.getId())
        s = s + "Physical Address: {} ".format(self.getPhysicalAddress())
        s = s + "Total Number of Data Points: {} ".format(self.getLength())
        s = s + "Attributes: {} ".format(self.getAttributes())
        s = s + "Min Values: {} ".format(self.min_values)
        s = s + "Max Values: {}".format(self.max_values)
        return s


class DatasetSplitter:

    def __init__(self):
        pass

    def split(self, dataset, attribute, value, idx, adjust=False):

        left = dataset.instance_values[0: idx]
        right = dataset.instance_values[idx:]
        lhs_set = Data(left, dataset.attr_types)
        rhs_set = Data(right, dataset.attr_types)

        # Set the Lhs min and Max and Rhs Max.
        lhs_set.calculate_limits()
        rhs_set.calculate_limits()

        self._splitNrVirtualPoints(dataset, attribute, value, lhs_set, rhs_set)

        if adjust:
            self._updateVirtualPoints(lhs_set)
            self._updateVirtualPoints(rhs_set)
        else:
            lhs_set.nr_total_instances = lhs_set.getLength() + lhs_set.nr_virtual_points
            rhs_set.nr_total_instances = rhs_set.getLength() + rhs_set.nr_virtual_points

        return lhs_set, rhs_set

    def _splitNrVirtualPoints(self, dataset, attribute, value, in_set, out_set):

        minV = dataset.get_min(attribute)
        maxV = dataset.get_max(attribute)

        in_set.nr_virtual_points = max(int(abs(dataset.nr_virtual_points * ((value - minV + 1) / (maxV - minV + 1)))),
                                       1)
        out_set.nr_virtual_points = dataset.nr_virtual_points - in_set.nr_virtual_points

        if out_set.nr_virtual_points < 0:
            self.raiseUndefinedNumberOfPoints()

        return

    def _updateVirtualPoints(self, data_set):

        nr_points_in_set = data_set.getLength()
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


class BuildTree_Iterative(object):
    def __init__(self, min_split, datasetDirectory='./', categoricalAttributes=[]):

        self.min_split = min_split
        self.root = None
        self.maxNodeId = 0
        self.datasetDirectory = datasetDirectory
        self.categoricalAttributes = categoricalAttributes
        self.cutCreator = InfoGainCutFactory(self.min_split)
        self.datasetSplitter = DatasetSplitter()

    def build(self, datasetPhysicalAddress):
        self._build_tree_iterative(datasetPhysicalAddress)
        return self.root

    def _build_tree_iterative(self, datasetPhysicalAddress):
        stack = list()
        parent = None
        datasetId = getRandomStringasId(0)
        depth = 0
        dataset = readPicklefile(datasetPhysicalAddress)
        DataSet_ParentChildMap = {}
        DataSet_ChildParentMap = {}
        CLNodes_IdToObjectMap = {}
        CLNodeIdToDataSetIdMap = {}
        DataSetIdToCLNodeIdMap = {}

        while True:
            if parent is not None:
                logger.info("At previous level({}), the cut happened on Attribute:{}, Value: {}".
                            format(parent.depth, parent.attribute, parent.cutValue))
            else:
                logger.info("This is the starting point!")

            bestCut = self._findBestCut(dataset)
            cutValue = None if bestCut is None else bestCut.value
            attribute = "" if bestCut is None else bestCut.attribute
            dataset.calculate_limits()
            attr_names = dataset.attr_names
            attr_idx = dataset.attr_idx
            max_values = dataset.max_values
            min_values = dataset.min_values
            length = dataset.getLength()
            curNodeId = self.maxNodeId
            dt_node = CLNode(curNodeId, datasetPhysicalAddress, datasetId, parent, attribute, depth, cutValue,
                             attr_names, attr_idx, max_values, min_values, length)

            CLNodes_IdToObjectMap[curNodeId] = dt_node
            CLNodeIdToDataSetIdMap[curNodeId] = datasetId
            DataSetIdToCLNodeIdMap[datasetId] = curNodeId
            self.maxNodeId += 1

            if parent:
                parent.addChildNode(dt_node)
            if self._isRootNode(depth):
                self.root = dt_node

            tapIntoRightDS = True

            if bestCut is not None:
                lhs_dataset, rhs_dataset = self._splitDatasetUsingBestCut(dataset, bestCut, adjust=True)
                parentDatasetId = datasetId
                datasetId = getRandomStringasId(curNodeId)
                L_datasetId = str(datasetId) + "_L"
                lhs_dataset.calculate_limits()

                R_datasetId = str(datasetId) + "_R"
                R_datasetPhysicalAddress = self.datasetDirectory + str(R_datasetId) + ".pickle"
                rhs_dataset.calculate_limits()
                writePickleFile(R_datasetPhysicalAddress, rhs_dataset)

                DataSet_ParentChildMap[parentDatasetId] = (L_datasetId, R_datasetId)
                DataSet_ChildParentMap[L_datasetId] = parentDatasetId
                DataSet_ChildParentMap[R_datasetId] = parentDatasetId

                stack.append(R_datasetId)

                if lhs_dataset.getLength() > self.min_split:
                    dataset = lhs_dataset
                    parent = dt_node
                    datasetId = L_datasetId
                    depth = depth + 1
                    tapIntoRightDS = False
                    datasetPhysicalAddress = None

                else:
                    L_datasetPhysicalAddress = self.datasetDirectory + str(L_datasetId) + ".pickle"
                    writePickleFile(L_datasetPhysicalAddress, lhs_dataset)

                    attr_names = lhs_dataset.attr_names
                    attr_idx = lhs_dataset.attr_idx
                    max_values = lhs_dataset.max_values
                    min_values = lhs_dataset.min_values
                    length = lhs_dataset.getLength()
                    curNodeId = self.maxNodeId
                    cNode = CLNode(curNodeId, L_datasetPhysicalAddress, L_datasetId, dt_node, None, depth + 1, None,
                                   attr_names, attr_idx, max_values, min_values, length)
                    dt_node.addChildNode(cNode)
                    CLNodes_IdToObjectMap[curNodeId] = cNode
                    CLNodeIdToDataSetIdMap[curNodeId] = L_datasetId
                    DataSetIdToCLNodeIdMap[L_datasetId] = curNodeId
                    self.maxNodeId += 1
            else:

                datasetPhysicalAddress = self.datasetDirectory + str(datasetId) + ".pickle"
                writePickleFile(data=dataset, path=datasetPhysicalAddress)
                dt_node.modifyUnderlyingDataSet(datasetId=datasetId, datasetPhysicalAddress=datasetPhysicalAddress)

            if tapIntoRightDS:
                if len(stack) > 0:
                    R_datasetId = stack.pop()
                    datasetId = R_datasetId
                    ParentDataSetId = DataSet_ChildParentMap[R_datasetId]
                    ParentNodeId = DataSetIdToCLNodeIdMap[ParentDataSetId]
                    parent = CLNodes_IdToObjectMap[ParentNodeId]
                    depth = parent.getDepth() + 1
                    R_datasetPhysicalAddress = self.datasetDirectory + R_datasetId + ".pickle"
                    datasetPhysicalAddress = R_datasetPhysicalAddress
                    dataset = readPicklefile(R_datasetPhysicalAddress)
                else:
                    break

    def _isRootNode(self, depth):
        if depth == 0 and self.root is None:
            return True

    def _splitDatasetUsingBestCut(self, dataset, bestCut, adjust):
        dataset.sort(bestCut.attribute)
        idx = dataset.getInstanceIndex(bestCut.inst_id)
        try:
            idx = idx[-1]
        except TypeError:
            pass

        lhs_set, rhs_set = self.datasetSplitter.split(dataset, bestCut.attribute,
                                                      bestCut.value, idx + 1, adjust)
        lhs_set.calculate_limits()
        rhs_set.calculate_limits()

        return lhs_set, rhs_set

    def _findBestCut(self, dataset):
        dataset.calculate_limits()
        bestCut = None
        if dataset.getLength() <= self.min_split:
            return bestCut

        for attribute in dataset.attr_names:
            dataset.sort(attribute)  # Sorting the dataset based on a attribute value at a time.
            di_cut1 = self._calcCut1(dataset, attribute)  # Best cut on current attribute based on "Information Gain"
            if di_cut1 is None:  # Ignore dimension go to the next dimension.
                continue
            if len(self.categoricalAttributes) > 0 and attribute in self.categoricalAttributes:
                bestCut = self._selectLowerDensityCut(di_cut1, bestCut)
                continue

            di_cut2 = self._calcCut2(di_cut1)  # Acquiring a second cut on the current attribute at Low Density Region.
            if di_cut2 is None:  # If there is no di_cut2 there, di_cut1 is the best cut on this dimension.
                bestCut = self._selectLowerDensityCut(di_cut1, bestCut)  # Keeping Track of best cut incrementally.
                continue  # We are done with this dimension, skip the rest and go to the next dimension.
            # Acquire a cut in the space which is in between di_cut1 and di_cut2.
            di_cut3 = self._calcCut3(di_cut1, di_cut2)
            if di_cut3 is None:  # if di_cut3 is no better than di_cut2 in terms of relative density the di_cut2 is the best cut on this dimesion.
                bestCut = self._selectLowerDensityCut(di_cut2, bestCut)  # Keeping Track of best cut incrementally.
            else:  # If there is a di_cut3 that means there is a lower density region and di_cut3 is the best cut on this dimension.
                bestCut = self._selectLowerDensityCut(di_cut3, bestCut)  # Keeping Track of best cut incrementally.
        return bestCut  # Returning the best cut at this level and the local dataset, which is a subset of the entire dataset.

    def _calcCut1(self, dataset, attribute):
        # Acquire the first cut on the dimension based on Information Gain
        return self.cutCreator.cut(dataset, attribute)

    def _calcCut2(self, di_cut1):
        # Acquire the lower realtive density region among the regions created by di_cut1
        lower_density_set, _ = di_cut1.getLowerDensityRegion()
        # Acquire the second cut on the dimension in the Lower Density Region based on Information Gain.
        return self.cutCreator.cut(lower_density_set, di_cut1.attribute)

    def _calcCut3(self, di_cut1, di_cut2):
        # Acquire at which region the value used to acquire the di_cut1 falls in as the result of di_cut2.
        adjacentRegion, otherRegion = di_cut2.getAdjacentRegion(di_cut1.value, di_cut1.attribute)
        di_cut3 = None
        # Compare relative densities among two regions as the result of di_cut2.
        if _relative_density(adjacentRegion) <= _relative_density(otherRegion):
            # Acquire the lower realtive density region among the regions created by di_cut2.
            lower_density_set, _ = di_cut2.getLowerDensityRegion()
            # Acquire the third cut on the dimension in the Lower Density Region based on Information Gain.
            di_cut3 = self.cutCreator.cut(lower_density_set, di_cut2.attribute)
        return di_cut3

    def _selectLowerDensityCut(self, cut1, cut2):
        if cut1 is None:
            return cut2
        if cut2 is None:
            return cut1
        rd1 = cut1.getRelativeDensityOfLowerDensityRegion()  # Acquire the lower density region by cut1
        rd2 = cut2.getRelativeDensityOfLowerDensityRegion()  # Acquire the lower density region by cut2
        if rd1 < rd2:
            return cut1  # Return the cut which has the lower density region with the lowest relative density.
        else:
            return cut2


class Cut:

    def __init__(self, attribute, value, inst_id, lhsset, rhsset):

        self.attribute = attribute
        self.value = value
        self.inst_id = inst_id
        self.lhsset = lhsset
        self.rhsset = rhsset
        self.lowerDensityRegion = None
        self.lowerDensity = None

    def __str__(self):
        s = 'Cut: ' + self.attribute + "\n"
        s += 'At: ' + str(self.value)
        return s

    def getAdjacentRegion(self, value, attribute):
        def getMinimumDistanceFromValue(dataset, attribute, value):
            distance1 = abs(dataset.get_max(attribute) - value)
            distance2 = abs(dataset.get_min(attribute) - value)
            return min(distance1, distance2)

        rhs_distance = getMinimumDistanceFromValue(self.rhsset, attribute, value)
        lhs_distance = getMinimumDistanceFromValue(self.lhsset, attribute, value)

        if lhs_distance < rhs_distance:
            return self.lhsset, self.rhsset
        else:
            return self.rhsset, self.lhsset

    def getRelativeDensityOfLowerDensityRegion(self):
        if self.lowerDensity is None:
            _, lowerDensity = self.getLowerDensityRegion()
            self.lowerDensity = lowerDensity

        return self.lowerDensity

    def getLowerDensityRegion(self):
        if self.lowerDensityRegion is None:
            if self.lhsset is None or self.rhsset is None:
                self.raiseNoRegionsDefined()

            rdL = _relative_density(self.lhsset)
            rdR = _relative_density(self.rhsset)
            if rdL > rdR:
                self.lowerDensityRegion = self.rhsset
                self.lowerDensity = rdR
            else:
                self.lowerDensityRegion = self.lhsset
                self.lowerDensity = rdL

        return self.lowerDensityRegion, self.lowerDensity

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
        di_cut_setting = {}
        max_info_gain = -1
        instances = dataset.getInstances(attribute)
        nunique = np.unique(instances)
        for value in nunique:
            i = np.argwhere(instances == value)
            i = i[-1]
            i = i[0]

            if self._hasRectangle(dataset, attribute, value):

                lhs_set, rhs_set = self.datasetSplitter.split(dataset, attribute, value, i + 1)
                ig = self._info_gain(dataset, lhs_set, rhs_set)
                if ig > max_info_gain:
                    max_info_gain = ig
                    di_cut_setting['attribute'] = attribute
                    di_cut_setting['value'] = value
                    di_cut_setting['id'] = dataset.getId(i)
                    di_cut_setting['index'] = i

        if max_info_gain > -1:
            lset, rset = self.datasetSplitter.split(dataset, attribute,
                                                    di_cut_setting['value'],
                                                    di_cut_setting['index'] + 1)
            di_cut = Cut(attribute, di_cut_setting['value'], dataset.getId(di_cut_setting['index']), lset, rset)

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

        if (lhs_set.nr_total_instances < 1) or (rhs_set.nr_total_instances < 1):
            return -1

        ratio_instances_lhs = (float(lhs_set.nr_total_instances) / dataset.nr_total_instances)
        ratio_instances_rhs = (float(rhs_set.nr_total_instances) / dataset.nr_total_instances)
        entropy2 = ratio_instances_lhs * self._calc_entropy(lhs_set) + ratio_instances_rhs * self._calc_entropy(rhs_set)

        entropy1 = self._calc_entropy(dataset)
        entropy = entropy1 - entropy2
        return entropy

    def _calc_entropy(self, dataset):

        nr_existing_instances = dataset.getLength()
        total = nr_existing_instances + dataset.nr_virtual_points
        terms = list()
        terms.append((float(nr_existing_instances) / float(total)) *
                     (np.log2(float(nr_existing_instances) / float(total))))
        terms.append((float(dataset.nr_virtual_points) / float(total)) *
                     (np.log2(float(dataset.nr_virtual_points) / float(total))))
        return sum(terms) * -1


class CLNode(object):
    # Added cut to it. So that It can be reused. Cut can be represented by value and attribute.
    def __init__(self, nodeId, datasetPhysicalAddress, datasetId, parent, attribute, depth, cutValue,
                 dsAttrNames, dsAttrIdx, maxValues, minValues, length):

        self.modifyUnderlyingDataSet(datasetId, datasetPhysicalAddress)
        self.datasetSummary = DataSetSummary(datasetId, datasetPhysicalAddress, dsAttrNames,
                                             dsAttrIdx, maxValues, minValues, length)
        self.parent = parent
        self.attribute = attribute
        self.cutValue = cutValue
        self.depth = depth
        self.children = list()
        self.can_prune = False
        self.includedInCluster = False
        self.clusterId = None
        self.nodeId = nodeId
        self.touchedNodes = None
        self.modifiedLength = None
        self.dataset = None

    def getDatasetPhysicalAddress(self):
        return self.datasetPhysicalAddress

    def getDatasetId(self):
        return self.datasetId

    def modifyUnderlyingDataSet(self, datasetId, datasetPhysicalAddress):
        if datasetPhysicalAddress is None:
            self._setDatasetId(datasetId)
            self._setDatasetPhysicalAddress(datasetPhysicalAddress)
            return True

        elif doThisPathExist(datasetPhysicalAddress):
            try:
                self._setDatasetId(datasetId)
                self._setDatasetPhysicalAddress(datasetPhysicalAddress)
                return True
            except RuntimeError:
                logger.error("Error! Unable to modify/assign the underlying dataset!")
                raise RuntimeError("Error! Unable to modify/assign the underlying dataset!")
                return False
        else:
            logger.error("The path, {}, doesn't exist or the program, doesn't have access to the path!".
                         format(datasetPhysicalAddress))
            raise FileNotFoundError("The path, {}, doesn't exist or the program, doesn't have access to the path!".
                                    format(datasetPhysicalAddress))
            return False

    def _setDatasetId(self, ii):
        self.datasetId = ii
        return True

    def _setDatasetPhysicalAddress(self, datasetPhysicalAddress):
        self.datasetPhysicalAddress = datasetPhysicalAddress
        return True

    def setPruneState(self, prune):
        self.can_prune = prune

    def isPrune(self):
        return self.can_prune

    def getRelativeDensity(self):
        dataset = readPicklefile(self.datasetPhysicalAddress)
        return _relative_density(dataset) * 100.0

    def getNrInstancesInNode(self):
        return self.datasetSummary.getLength()

    def addChildNode(self, node):
        self.children.append(node)

    def getChildNodes(self):
        return self.children

    def getLeft(self):
        if len(self.children) == 0:
            return None
        else:
            return self.children[0]

    def getRight(self):
        if len(self.children) < 2:
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

    def getId(self):
        return self.nodeId

    def getTouchedNodes(self):
        return self.touchedNodes

    def addTouchedNodes(self, nodeId):
        if self.touchedNodes is None:
            self.touchedNodes = list()

        self.touchedNodes.append(nodeId)
        return

    def getDepth(self):
        return self.depth

    def setModifiedDataLength(self, modifiedLength):
        self.modifiedLength = modifiedLength
        return

    def getModifiedDataLength(self):
        return self.modifiedLength

    def getDatasetSummary(self):
        return self.datasetSummary

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
        s += str(self.getNrInstancesInNode()) + ' instances, '
        s += ", " + str(int(self.getRelativeDensity())) + " relative density " + '\n'
        s += ", " + str(self.depth) + " depth " + '\n'
        s += "Cuts " + str(attr) + " At Values " + str(vals) + '\n'
        return s

    def str2(self):
        s = 'ID: {}'.format(self.nodeId)
        s += ' Inst: {} '.format(self.getNrInstancesInNode())
        s += "Depth: {}".format(self.getDepth())
        s += '' if self.clusterId is None else self.clusterId
        return s


class CLTree:
    def __init__(self, dataset, min_split=100, tmpFilesPath='./'):
        if dataset is None:
            self.raiseUndefinedDataset()

        self.min_split = min_split
        self.clusters = dict()
        self.nodes = dict()
        self.clustersStatistics = None
        self.root = None

        tmpStoringPath = tmpFilesPath + "CLTree/"
        if doThisPathExist(tmpStoringPath):
            raise RuntimeError ("Path already exists!")
        else:
            createPath(tmpStoringPath)
            self.storageDirectory = tmpStoringPath

        datasetDirectory = self.storageDirectory + "DataSet/"
        self.datasetDirectory = datasetDirectory
        if not doThisPathExist(self.datasetDirectory):
            createPath(self.datasetDirectory)
        dataset.calculate_limits()
        categoricalAttributes = []
        for attr in dataset.attr_names:
            instances = dataset.getInstances(attr)
            nunique = np.unique(instances)
            if len(nunique) < 3:
                categoricalAttributes.append(attr)
        self.categoricalAttributes = categoricalAttributes

        self.datasetPhysicalAddress = self.datasetDirectory + 'DataSet_' + 'Original' + '.pickle'
        writePickleFile(path=self.datasetPhysicalAddress, data=dataset)

    def buildTree(self):
        b = BuildTree_Iterative(self.min_split, self.datasetDirectory, self.categoricalAttributes)
        self.root = b.build(self.datasetPhysicalAddress)
        allNodes = self._collectAllNodes(self.root, dict())
        self.nodes = allNodes
        return

    def getDataset(self):
        dataset = readPicklefile(self.datasetPhysicalAddress)
        dataset.calculate_limits()
        return dataset

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
            nodeDataset = readPicklefile(node.datasetPhysicalAddress)
            nodeDataset.calculate_limits()
            node.dataset = nodeDataset
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
