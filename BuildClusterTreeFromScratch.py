from CLTreeModules_IterativeCompatible import CLTree, DataFrameReader
from ReadWritePickleFile import writePickleFile
from PruneTreeByMergingCentroids_IterativeCompatiable import pruneByGridSearch_Centroid
from PruneTreeByConqueringNodes_IterativeCompatiable import pruneByGridSearch
from CLTree_Utility import doThisPathExist, createPath, getRandomStringasId
from ClusterTree_Utility import getAttributesAndClusters
from Attribute import Attribute
from ClusterTree import ClusterTreeNode, ClusterTree
from BasicDataPrep_V1 import getData, getOthersGoup
from DataPrepForSeasonalityDetectionUsingBookingTrend_V1 import bucketizeLeadDays, \
    dataPrepForClusteringByBookingTrend, readSeparateSeasonalityDetectionData
from SeasonalityDetection_Discrete import ClusterMonths_Discrete
from SeasonalityDetection_Continuous import ClusterMonths_Continuous
from ClusteringFromRules import getClusterTreeFromRules
import pandas as pd
import numpy as np
import datetime
import logging
import math
import re

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def getSimpleAttributes(cltree, othersGroup):
    # Assert the Tree is an instance of CLTree
    assert isinstance(cltree, CLTree)
    dataset = cltree.getDataset()
    dataset.calculate_limits()
    attributes = {}
    i = 0
    while i < len(dataset.attr_names):
        attr_name = dataset.attr_names[i]
        uniqueValues = len(np.unique(dataset.getInstances(attr_name)))

        attributes[dataset.attr_names[i]] = Attribute(name=attr_name,
                                                      noOfUniqueValues=uniqueValues,
                                                      maxVal=dataset.max_values[i + 1],
                                                      minVal=dataset.min_values[i + 1])

        i += 1

    for key in attributes.keys():
        if attributes[key].type == 'Calculated' and re.findall("_others$", attributes[key].name) \
                and attributes[key].originalAttribute in othersGroup:
            attributes[key].setOriginalAttributeVal(othersGroup[attributes[key].originalAttribute])
        elif attributes[key].type == 'Calculated' and re.findall("_weekend$", attributes[key].name):
            attributes[key].setOriginalAttributeVal(['friday', 'saturday', 'fri', 'sat'])
        elif attributes[key].type == 'Calculated' and re.findall("_midweek$", attributes[key].name):
            attributes[key].setOriginalAttributeVal(['tuesday', 'wednesday', 'thursday', 'tue', 'wed', 'thu'])
        elif attributes[key].type == 'Calculated' and re.findall("_earlyweek$", attributes[key].name):
            attributes[key].setOriginalAttributeVal(['sunday', 'monday', 'sun', 'mon'])

    return attributes


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


def __validateMinCriteria(min_y, min_split_fraction, data_length,
                          fractionOfTotalData, mergingCentroidsVsconqueringNodes):
    assert 0. < fractionOfTotalData <= 1.0, \
        logger.error("The value of 'fractionOfTotalData' should be greater than 0 and less than or equal to 1.")

    if fractionOfTotalData < 1.0:
        min_y = math.floor(min_y * fractionOfTotalData)

    if not mergingCentroidsVsconqueringNodes:
        min_split_fraction = max(min(min_split_fraction, 0.001), (1. / data_length))
    else:
        min_split_fraction = max(min(min_split_fraction, (min_y / data_length)), (1. / data_length))

    min_y = max(min_y, math.ceil(min_split_fraction * data_length))

    logger.info("The min-split fraction has been set to {}".format(min_split_fraction))
    logger.info("The min_y (The minimum number of member required for a cluster) has been set to {}".format(min_y))

    return min_y, min_split_fraction


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


def __getClusterTreeFromData(data, categoricalAttributes, mergingCentroidsVsconqueringNodes, balancedPrune,
                             min_y, min_split_fraction, conquerDataColumns, prefixString, defaultValues,
                             useSilhouette, prevVer, tmpFilesPath, cltreeSpecificUtility, subFolderString = ''):
    if categoricalAttributes is not None:
        othersGroup, data = getOthersGoup(data, categoricalAttributes, min_split_fraction)
    else:
        othersGroup = None

    data.reset_index(inplace=True, drop=True)

    if not mergingCentroidsVsconqueringNodes:
        data['rowNoAsId'] = data.index
        if conquerDataColumns is not None:
            conquerData = pd.DataFrame(data[conquerDataColumns + ['rowNoAsId']].copy(deep=True))
            data = data[list(set(data.columns) - set(conquerDataColumns))].copy(deep=True)
        else:
            conquerData = data.copy(deep=True)

        divideData = data.copy(deep=True)

        data.drop(columns='rowNoAsId', inplace=True)
    else:
        divideData = data.copy(deep=True)

    d_var = dict(data.var())
    d_cols = sorted(d_var, key=d_var.get, reverse=True)
    data = data[d_cols]

    r = DataFrameReader(data)
    data = r.read()
    min_split = np.ceil(data.getLength() * min_split_fraction)
    cltree = CLTree(data, min_split, tmpFilesPath)
    cltree.buildTree()

    pathToStoreModel = tmpFilesPath + "CLTree/" + "Model/"
    if not doThisPathExist(pathToStoreModel):
        createPath(pathToStoreModel)

    # store the model for later use
    pathToStoreModel = pathToStoreModel + prefixString + "CLTree.pickle"
    relativePathToStoreModel = subFolderString + "CLTree/" + "Model/" + prefixString + "CLTree.pickle"
    writePickleFile(pathToStoreModel, cltree)
    refUtility = {}
    refUtility['CLTreeModelPath'] = relativePathToStoreModel

    min_y = max(min_y, min_split)
    data_length = data.getLength()

    pathToStoreData = tmpFilesPath + "CLTree/" + "DataForPrune/"
    if not doThisPathExist(pathToStoreData):
        createPath(pathToStoreData)

    if mergingCentroidsVsconqueringNodes:
        result, baseVer = pruneByGridSearch_Centroid(cltree, min_y, data_length, prefixString,
                                                     balancedPrune, useSilhouette, tmpFilesPath)
        divideDataPath = pathToStoreData + prefixString + "DivideData.csv"
        divideData.to_csv(divideDataPath, sep=";")
        relativePathToDivideData = subFolderString + "CLTree/" + "DataForPrune/" + prefixString + "DivideData.csv"
        refUtility['DivideDataPath'] = relativePathToDivideData
        refUtility['ConquerDataPath'] = relativePathToDivideData

    else:

        # store the divide and conquer data
        divideDataPath = pathToStoreData + prefixString + "DivideData.csv"
        divideData.to_csv(divideDataPath, sep=";")
        relativePathToDivideData = subFolderString + "CLTree/" + "DataForPrune/" + prefixString + "DivideData.csv"
        refUtility['DivideDataPath'] = relativePathToDivideData

        conquerDataPath = pathToStoreData + prefixString + "ConquerData.csv"
        conquerData.to_csv(conquerDataPath, sep=";")
        relativePathToConquerData = subFolderString + "CLTree/" + "DataForPrune/" + prefixString + "ConquerData.csv"
        refUtility['ConquerDataPath'] = relativePathToConquerData
        # Include RowAsId in Prune instead of index or make RowAsId index.

        gradientTolerance = 0.01  # Tested with different values, looks like 0.01 is a good candidate
        result, baseVer = pruneByGridSearch(cltree, min_y, prefixString, gradientTolerance, conquerData,
                                            divideData, useSilhouette, tmpFilesPath)

    utc_now = datetime.datetime.now()
    deltaVer = math.ceil(float((utc_now - datetime.datetime(1970, 1, 1)).total_seconds()))
    deltaVer = float("." + str(deltaVer))

    if math.floor(prevVer) == math.floor(baseVer):
        curVer = baseVer + deltaVer
    else:
        curVer = baseVer

    attributes = getSimpleAttributes(cltree=cltree, othersGroup=othersGroup)
    cltreeSpecificUtility['othersGroup'] = othersGroup
    cltreeSpecificUtility['attributes'] = attributes

    pathToCltreeSpecificUtility = tmpFilesPath + "CLTree/" + "Utility/"
    if not doThisPathExist(pathToCltreeSpecificUtility):
        createPath(pathToCltreeSpecificUtility)
    pathToCltreeSpecificUtility = pathToCltreeSpecificUtility + prefixString + "Utility.pickle"
    writePickleFile(pathToCltreeSpecificUtility, cltreeSpecificUtility)
    relativePathToCltreeSpecificUtility = subFolderString + "CLTree/" + "Utility/" + prefixString + "Utility.pickle"
    refUtility['CltreeSpecificUtilityPath'] = relativePathToCltreeSpecificUtility

    simpleTreeRoot = buildSimpleTree(clnode=cltree.getRoot(), node=None, attributes=attributes)
    clusterTree = ClusterTree(simpleTreeRoot, defaultValues, curVer)
    return clusterTree, refUtility


def buildSimpleClusterTree(option=1, destinationPath=None, staticClusterFilePath=None,
                           storeID=None, histrocalDataFilePath=None, categoricalAttributes=None,
                           manualSeasonality=None, seasonalityDataPath=None, continuousVsDiscrete=True,
                           keyStringForSeasonality='arrival', mergingCentroidsVsconqueringNodes=True,
                           balancedPrune=False, conquerDataColumns=None,
                           min_y=None, min_split_fraction=None, fractionOfTotalData=None,
                           previousVersion=-1, missingDataTolerance=0.90, delimiter=';',
                           useSilhouette=False, tmpFilesPath='./'):
    if min_y is None:
        min_y = 1000.0
    if min_split_fraction is None and option == 2:
        min_split_fraction = 0.001
    elif min_split_fraction is None and option == 3:
        min_split_fraction = 0.010

    if fractionOfTotalData is None:
        fractionOfTotalData = 1.

    assert 0. < fractionOfTotalData <= 1., \
        logger.error("The value of 'fractionOfTotalData' should be between 0(exclusive) and 1(inclusive)")
    assert destinationPath is not None, logger.error("Please provide a destination path to save the result!")
    assert (option == 1 and staticClusterFilePath is not None) or \
           (option in (2, 3) and histrocalDataFilePath is not None), \
        logger.error("Please provide the appropriate source file path and try again!")

    assert option in (1, 2, 3), logger.error("Please provide correct value as option. Acceptable values are 1,2,3!")

    if previousVersion is None:
        previousVersion = -1

    if option == 1:

        clusterTree = getClusterTreeFromRules(staticClusterFilePath, previousVersion)
        generalUtilityPath = None

    elif option in (2, 3):
        essentials = {}
        essentials['option'] = option
        essentials['fractionOfTotalData'] = fractionOfTotalData

        assert tmpFilesPath is not None and doThisPathExist(tmpFilesPath), \
            logger.error("Please provide a valid path using 'tmpFilesPath'")


        if categoricalAttributes is not None:
            if isinstance(categoricalAttributes, list):
                categoricalAttributes = [i.lower().strip() for i in categoricalAttributes]
            elif isinstance(categoricalAttributes, str):
                categoricalAttributes = categoricalAttributes.lower().strip()

        if conquerDataColumns is not None:
            assert isinstance(conquerDataColumns, list), "'conquerDataColumns' needed to be provided in a list format!"
            conquerDataColumns = [i.lower().strip() for i in conquerDataColumns]

        else:
            if not mergingCentroidsVsconqueringNodes:
                logger.warning("As 'conquerDataColumns' has not been provided,\
                                the algorithm will reuse the data used for creating the initial tree!")

        if keyStringForSeasonality is not None:
            keyStringForSeasonality = keyStringForSeasonality.strip().lower()
            essentials['keyStringForSeasonality'] = keyStringForSeasonality

        df, keyStringDateCol, keyStringWeekdayCol, keyStringMonthCol, defaultValues = \
            getData(histrocalDataFilePath, categoricalAttributes, keyStringForSeasonality,
                    conquerDataColumns, missingDataTolerance, delimiter=delimiter)
        essentials['totalRecordCount'] = len(df)

        essentials['keyStringDateCol'] = keyStringDateCol
        essentials['keyStringWeekdayCol'] = keyStringWeekdayCol
        essentials['keyStringMonthCol'] = keyStringMonthCol

        essentials['defaultValues'] = defaultValues

        if categoricalAttributes is not None and isinstance(categoricalAttributes, list) \
                and len(categoricalAttributes) > 0:

            assert (set(categoricalAttributes).issubset(set(df.columns))), \
                logger.error("The name of categorical attributes don't match with data column names!")

        elif categoricalAttributes is not None and isinstance(categoricalAttributes, str):
            assert categoricalAttributes in list(df.columns), \
                logger.error("The name of categorical attribute doesn't match with data column names!")

        essentials['categoricalAttributes'] = categoricalAttributes

        while True:
            tmpRunningId = getRandomStringasId()
            tmpStoringPath = tmpFilesPath + "RawClusterModel_" + str(tmpRunningId)+ "_" +\
                             str(len(df)) + "_" + "{:.2E}".format(min_split_fraction) + "/"
            if doThisPathExist(tmpStoringPath):
                continue
            else:
                createPath(tmpStoringPath)
                ModelPathUtility = tmpStoringPath
                break

        if option == 3:
            if manualSeasonality is not None:
                assert isinstance(manualSeasonality, dict) and \
                       all(isinstance(i, list) for i in list(manualSeasonality.values())) and \
                       len(manualSeasonality) <= 3 and len(
                    [j for i in list(manualSeasonality.values()) for j in i]) == 12 \
                       and max([j for i in list(manualSeasonality.values()) for j in i]) == 12 and \
                       max([[j for i in list(manualSeasonality.values()) for j in i].count(x) for x in
                            set([j for i in list(manualSeasonality.values()) for j in i])]) == 1, \
                       logger.error("Please provide manual seasonality in correct format! The 'manualSeasonality'\
                                takes only 'dict'/key-value pair, where values are list of numerical value of months!\
                                One month can appear in only one season but one season can have any number of months!")

                clusters = manualSeasonality
                essentials['continuousVsDiscrete'] = None

            else:
                essentials['continuousVsDiscrete'] = continuousVsDiscrete

                if seasonalityDataPath is not None:
                    assert keyStringMonthCol is not None, logger.error("To apply seasonality to data\
                                            we would need to provide **Relevant Month related information!")
                    record = readSeparateSeasonalityDetectionData(sourcePath=seasonalityDataPath, delim=delimiter)

                else:
                    assert keyStringWeekdayCol is not None and keyStringMonthCol is not None, \
                        logger.error("To detect seasonality we would need to provide"
                                     " **Relevant Month and Weekday related information!")
                    seasonalityData = df.copy(deep=True)
                    seasonalityData, _map, bucketColName = bucketizeLeadDays(seasonalityData,
                                                                             leadDaysColName='leaddays')
                    record = dataPrepForClusteringByBookingTrend(seasonalityData, _map, keyStringDateCol,
                                                                 keyStringWeekdayCol, keyStringMonthCol,
                                                                 bucketColName='LeadDays_Bucket',
                                                                 groupByHowDissimilarToOthersTo=False)

                if continuousVsDiscrete:  # True means we have been asked to provide continuous seasonality.
                    if keyStringDateCol is not None:
                        df_copy = df.copy(deep=True)
                        instanceCountByMonth = df_copy[[keyStringDateCol, keyStringMonthCol]]. \
                            drop_duplicates().reset_index(drop=True)
                        del df_copy
                        instanceCountByMonth = dict(instanceCountByMonth[keyStringMonthCol].value_counts())
                    else:
                        instanceCountByMonth = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31,
                                                11: 30, 12: 31}

                    clusters = ClusterMonths_Continuous(ds_EachRowADataPoint=record,
                                                        instanceCountByEachDataPoint=instanceCountByMonth,
                                                        acceptableNumberOfClusters=[4, 5, 6])
                else:  # False means if is okay to have group of months which are not adjacent to each other.
                    clusters = ClusterMonths_Discrete(record, plot=False)

                del record

            if keyStringDateCol is not None:
                df.drop(columns=keyStringDateCol, inplace=True)
            logger.info("Cluster of months: {}".format(clusters))

            essentials['seasonsByMonths'] = clusters

            monthsToTree = {}
            curVer = previousVersion
            refUtilities = {}
            for c in clusters:
                cltreeSpecificUtility = {}

                prefixString = ''.join([str(i) for i in clusters[c]])
                prefixString = str(storeID) + '_' + prefixString + '_'
                cltreeSpecificUtility['prefixString'] = prefixString

                data = pd.DataFrame(df[df[keyStringMonthCol].isin(clusters[c])])
                data.drop(columns=keyStringMonthCol, inplace=True)

                subFolderString = 'Season_' + keyStringMonthCol + '_' + ''.join([str(m) for m in clusters[c]]) + "/"
                seasonSubFolder = ModelPathUtility + subFolderString
                try:
                    createPath(seasonSubFolder)
                except PermissionError:
                    raise PermissionError ("Couldn't create folder to store files!")

                dataLength = len(data)
                cltreeSpecificUtility['dataLength'] = dataLength

                min_y, min_split_fraction = __validateMinCriteria(min_y, min_split_fraction,
                                                                  dataLength, fractionOfTotalData,
                                                                  mergingCentroidsVsconqueringNodes)
                cltreeSpecificUtility['min_split_fraction'] = min_split_fraction

                tree, refUtility = __getClusterTreeFromData(data, categoricalAttributes,
                                                            mergingCentroidsVsconqueringNodes,
                                                            balancedPrune, min_y, min_split_fraction,
                                                            conquerDataColumns,
                                                            prefixString, defaultValues,
                                                            useSilhouette, previousVersion,
                                                            seasonSubFolder, cltreeSpecificUtility, subFolderString)

                monthsToTree[tuple(clusters[c])] = tree

                refUtilities[tuple(clusters[c])] = refUtility

                curVer = max(curVer, tree.versionOfClusterAlgo)

            attribute = keyStringMonthCol.strip().lower()
            keys = list(monthsToTree.keys())
            clusterTree = __combine(monthsToTree, keys, attribute, defaultValues, curVer)
            essentials['RefUtilities'] = refUtilities
            essentials['Version'] = curVer
        else:
            cltreeSpecificUtility = {}
            prefixString = str(storeID) + '_'
            cltreeSpecificUtility['prefixString'] = prefixString

            if keyStringDateCol is not None:
                df.drop(columns=keyStringDateCol, inplace=True)
            if keyStringMonthCol is not None:
                df.drop(columns=keyStringMonthCol, inplace=True)

            dataLength = len(df)
            cltreeSpecificUtility['dataLength'] = dataLength

            min_y, min_split_fraction = __validateMinCriteria(min_y, min_split_fraction, dataLength,
                                                              fractionOfTotalData,
                                                              mergingCentroidsVsconqueringNodes)

            cltreeSpecificUtility['min_split_fraction'] = min_split_fraction

            clusterTree, refUtility = __getClusterTreeFromData(df, categoricalAttributes,
                                                               mergingCentroidsVsconqueringNodes,
                                                               balancedPrune, min_y, min_split_fraction,
                                                               conquerDataColumns,
                                                               prefixString, defaultValues, useSilhouette,
                                                               previousVersion,
                                                               ModelPathUtility, cltreeSpecificUtility)

            essentials['RefUtilities'] = refUtility
            essentials['Version'] = clusterTree.versionOfClusterAlgo

        generalUtilityPath = ModelPathUtility + 'essentials.pickle'
        writePickleFile(path=generalUtilityPath, data=essentials)

    attributes, clusters = getAttributesAndClusters(clusterTree)
    flag = writePickleFile(path=destinationPath, data=clusterTree)
    if not flag:
        logger.error("The process has failed! Please Try Again!")
        return None
    else:
        logger.info("The process is successful!")
        return attributes, clusters, generalUtilityPath
