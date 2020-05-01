from ReadWritePickleFile import readPicklefile, writePickleFile
from PruneTreeByMergingCentroids_IterativeCompatiable import pruneByGridSearch_Centroid
from PruneTreeByConqueringNodes_IterativeCompatiable import pruneByGridSearch
from ClusterTree_Utility import getAttributesAndClusters
from ClusterTree import ClusterTreeNode, ClusterTree
import math
import pandas as pd
import datetime
import logging
import os

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def buildSimpleTree(clnode, node, attributes):
    # Check if ClusterTreeNode is None and clnode is Root
    curNode = ClusterTreeNode()

    if clnode.includedInCluster:
        curNode.setParent(node)
        if clnode.getParent() is None:
            curNode.setInheriatedFraction(1.0)
            if clnode.clusterId is not None:
                curNode.setClusterId(clnode.clusterId.strip().lower())
            else:
                curNode.setClusterId('DEFAULT')
        else:
            curDataLength = clnode.getDatasetSummary().getLength()
            parentDataLength = clnode.getParent().getDatasetSummary().getLength()
            curNode.setInheriatedFraction(curDataLength / parentDataLength)
            curNode.setClusterId(clnode.clusterId.strip().lower())

    else:
        attribute = clnode.attribute
        cutValue = clnode.cutValue

        if attributes[attribute].type == 'Categorical' or attributes[attribute].type == 'Calculated':
            curNode.setAttribute(attribute=attributes[attribute].originalAttribute,
                                 attributeType=attributes[attribute].type)
            curNode.setValue(attributes[attribute].originalAttributeVal)

        else:
            curNode.setAttribute(attribute=attributes[attribute].originalAttribute,
                                 attributeType=attributes[attribute].type)
            curNode.setValue(cutValue)

        if clnode.depth > 0:
            curDataLength = clnode.getDatasetSummary().getLength()
            parentDataLength = clnode.getParent().getDatasetSummary().getLength()
            curNode.setInheriatedFraction(curDataLength / parentDataLength)

        else:
            curNode.setInheriatedFraction(0.)

        curNode.setParent(node)
        CLNodeChildren = clnode.getChildNodes()
        left = buildSimpleTree(CLNodeChildren[0], curNode, attributes)
        curNode.setLeft(left)
        right = buildSimpleTree(CLNodeChildren[1], curNode, attributes)
        curNode.setRight(right)

    return curNode


def __combine(monthsToTree, keys, attribute, defaultValues, curVer):
    if len(keys) == 1:
        key = keys[0]
        return monthsToTree[key]

    curNode = ClusterTreeNode()
    mid = math.ceil((len(keys) - 1) / 2)
    keys_l = keys[:mid]
    keys_r = keys[mid:]
    subtree_l = __combine(monthsToTree, keys_l, attribute, defaultValues, curVer)
    subtree_r = __combine(monthsToTree, keys_r, attribute, defaultValues, curVer)
    cutValue = ()
    for i in keys_r:
        cutValue = cutValue + i
    curNode.setAttribute(attribute=attribute, attributeType='Calculated')
    curNode.setValue(list(cutValue))
    curNode.setLeft(subtree_l.getRoot())
    subtree_l.getRoot().setParent(curNode)
    curNode.setRight(subtree_r.getRoot())
    subtree_r.getRoot().setParent(curNode)
    clusterTree = ClusterTree(curNode, defaultValues, curVer)
    return clusterTree


def __validateMinYCriteria(min_y, min_split_fraction, data_length, fractionOfTotalData):
    assert 0. < fractionOfTotalData <= 1.0, \
        logger.error("The value of 'fractionOfTotalData' should be greater than 0 and less than or equal to 1.")

    if fractionOfTotalData < 1.0:
        min_y = math.floor(min_y * fractionOfTotalData)

    min_y = max(min_y, math.ceil(min_split_fraction * data_length))

    logger.info("The min_y (The minimum number of member required for a cluster) has been set to {}".format(min_y))

    return min_y


def __getClusterTreeFromPrebuiltCLTree(mergingCentroidsVsconqueringNodes, balancedPrune,
                                       min_y, defaultValues, useSilhouette,
                                       prevVer, tmpFilesPath, clTreeModelPath, clTreeModelUtilityPath,
                                       divideDataPath, conquerDataPath, delimiter=';', fractionOfTotalData=1.0):
    cltree = readPicklefile(clTreeModelPath)
    cltreeSpecificUtility = readPicklefile(clTreeModelUtilityPath)

    min_split_fraction = cltreeSpecificUtility.get('min_split_fraction')
    data_length = cltreeSpecificUtility.get('dataLength')
    min_y = __validateMinYCriteria(min_y, min_split_fraction, data_length, fractionOfTotalData)
    prefixString = cltreeSpecificUtility.get('prefixString')

    if mergingCentroidsVsconqueringNodes:
        result, baseVer = pruneByGridSearch_Centroid(cltree, min_y, data_length, prefixString, balancedPrune,
                                                     useSilhouette, tmpFilesPath)
    else:
        conquerData = pd.read_csv(conquerDataPath, delimiter=delimiter)
        for col in conquerData.columns:
            if col == 'Unnamed: 0' or col == 'unnamed: 0':
                conquerData.drop(columns=col, inplace=True)

        divideData = pd.read_csv(divideDataPath, delimiter=delimiter)
        for col in divideData.columns:
            if col == 'Unnamed: 0' or col == 'unnamed: 0':
                divideData.drop(columns=col, inplace=True)

        gradientTolerance = 0.01  # Tested with different values, looks like 0.01 is a good candidate
        result, baseVer = pruneByGridSearch(cltree, min_y, prefixString, gradientTolerance, conquerData, divideData,
                                            useSilhouette, tmpFilesPath)

    utc_now = datetime.datetime.now()
    deltaVer = math.ceil(float((utc_now - datetime.datetime(1970, 1, 1)).total_seconds()))
    deltaVer = float("." + str(deltaVer))

    if math.floor(prevVer) == math.floor(baseVer):
        curVer = baseVer + deltaVer
    else:
        curVer = baseVer

    attributes = cltreeSpecificUtility['attributes']

    simpleTreeRoot = buildSimpleTree(clnode=cltree.getRoot(), node=None, attributes=attributes)
    clusterTree = ClusterTree(simpleTreeRoot, defaultValues, curVer)
    return clusterTree


def createClusterTreeFromPrebuiltCLTree(prebuiltCLTreeUtilityPath, destinationPath,
                                        mergingCentroidsVsconqueringNodes=True,
                                        balancedPrune=False, min_y=None,
                                        useSilhouette=False, prevVer=1.0, tmpFilesPath='./'):
    assert destinationPath is not None
    cltreeGeneralUtility = readPicklefile(prebuiltCLTreeUtilityPath)
    dirName = os.path.dirname(prebuiltCLTreeUtilityPath)

    option = cltreeGeneralUtility.get('option')
    fractionOfTotalData = cltreeGeneralUtility.get('fractionOfTotalData')
    defaultValues = cltreeGeneralUtility.get('defaultValues')
    delimiter = ';'

    if option == 3:
        curVer = -1
        monthsToCLTree = cltreeGeneralUtility['RefUtilities']
        monthsToTree = {}
        for months in monthsToCLTree.keys():
            clTreeModelPath = monthsToCLTree[months].get('CLTreeModelPath')
            clTreeModelPath = os.path.join(dirName, clTreeModelPath)
            clTreeModelUtilityPath = monthsToCLTree[months].get('CltreeSpecificUtilityPath')
            clTreeModelUtilityPath = os.path.join(dirName, clTreeModelUtilityPath)
            divideDataPath = monthsToCLTree[months].get('DivideDataPath', None)
            if divideDataPath is not None:
                divideDataPath = os.path.join(dirName, divideDataPath)
            conquerDataPath = monthsToCLTree[months].get('ConquerDataPath', None)
            if conquerDataPath is not None:
                conquerDataPath = os.path.join(dirName, conquerDataPath)
            tree = __getClusterTreeFromPrebuiltCLTree(mergingCentroidsVsconqueringNodes, balancedPrune, min_y,
                                                      defaultValues, useSilhouette, prevVer, tmpFilesPath,
                                                      clTreeModelPath, clTreeModelUtilityPath, divideDataPath,
                                                      conquerDataPath, delimiter, fractionOfTotalData)

            curVer = max(curVer, tree.versionOfClusterAlgo)
            monthsToTree[tuple(months)] = tree

        keyStringMonthCol = cltreeGeneralUtility['keyStringMonthCol']
        assert keyStringMonthCol is not None
        attribute = keyStringMonthCol.strip().lower()
        keys = list(monthsToTree.keys())
        assert defaultValues is not None
        clusterTree = __combine(monthsToTree, keys, attribute, defaultValues, curVer)

    elif option == 2:
        curVer = -1
        clTreeRelatedInformation = cltreeGeneralUtility['RefUtilities']
        clTreeModelPath = clTreeRelatedInformation.get('CLTreeModelPath')
        clTreeModelPath = os.path.join(dirName, clTreeModelPath)
        clTreeModelUtilityPath = clTreeRelatedInformation.get('CltreeSpecificUtilityPath')
        clTreeModelUtilityPath = os.path.join(dirName, clTreeModelUtilityPath)
        divideDataPath = clTreeRelatedInformation.get('DivideDataPath', None)
        if divideDataPath is not None:
            divideDataPath = os.path.join(dirName, divideDataPath)
        conquerDataPath = clTreeRelatedInformation.get('ConquerDataPath', None)
        if conquerDataPath is not None:
            conquerDataPath = os.path.join(dirName, conquerDataPath)

        clusterTree = __getClusterTreeFromPrebuiltCLTree(mergingCentroidsVsconqueringNodes, balancedPrune,
                                                         min_y, defaultValues, useSilhouette,
                                                         prevVer, tmpFilesPath, clTreeModelPath, clTreeModelUtilityPath,
                                                         divideDataPath, conquerDataPath, delimiter,
                                                         fractionOfTotalData)

    else:
        raise RuntimeError("The Data Got Corrupted!")

    attributes, clusters = getAttributesAndClusters(clusterTree)

    flag = writePickleFile(path=destinationPath, data=clusterTree)
    if not flag:
        logger.error("The process has failed! Please Try Again!")
        return None
    else:
        logger.info("The process is successful!")
        return attributes, clusters
