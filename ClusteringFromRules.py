"""
1. Expect the rules to be provided in a .json format. [Consult DemoRuleBasedSegmentation.xlsx and
writePickleFile.ipynb/.html] to understand how the data is presented in static_clustering.pickle.
 This program will read static_clustering.pickle and do the further work as needed.

2. Every variable/feature/attribute required to provide a range to dictate the various values it can take.For
 categorical variables it must a list of all the values the underlying variable can take. For numerical variables the
 job can be done by providing Min and Max.

3. 1st step is validating all the cluster conditions. It checks for if all the variables mentioned in all the cluster
 rules of all the clusters are of expected data type and within the range or set of expected values.

4. Based on the cluster conditions we curate a data-set for the decision tree to get trained on. Consult,
 curatedData.csv to get more idea.

5. Validate the exclusivity of the cluster conditions by accuracy of the "DecisionTreeClassifier". If your cluster
 condtions are mutually exclusive and collectively exhaustive, the accuracy would be 1. or 100%.

6. At this moment we construct a list of instances of [Attribute Class] attributes from the tree returned by
 "DecisionTreeClassifier".

7. Once we constructed our list of instances of [Attribute Class] , we traverse the tree returned by
 "DecisionTreeClassifier" to build our simple "ClusterTree" and then assign default conditions and expected data type
  for each attribute. At the end we store the "ClusterTree" for future use in ".pickle" format.


Git Hub Link of DecisionTreeClassifier: https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/tree/tree.py

We are just re-purposing the original "DecisionTreeClassifier" algorithm.
Reusing 6 attributes of the original tree returned by "DecisionTreeClassifier" after it got trained on our curated data.

The attributes we are using are, threshold, feature, value and in addition one known flag value , -2.

1. -2 is flag to denote "leaf" or terminal node at original decision tree.
2. "threshold" is a 1D array. Each index is a proxy Node Id. At each index the content of this array denotes
    what is the "deciding" value at current node unless the value is -2. In that case it is the leaf node. There is
    no further path on this route.
3. "value" is a 2D array. The first index is a proxy Node Id. Here the tree store frequency/count of each "calss"es
    at current node. As our conditions has to be "Mutually Exclusive and Collectively Exhaustive" all the leaf nodes
    contains data from just one class/cluster. Target classes are sorted alphabetically.
4. Like "threshold", "feature" is also a 1-D array. Each index is a proxy Node Id. Unless it is a leaf node(denoted by
     -2), at each index the content of this array holds index of the feature on which the "deciding value mentioned at
          "threshold" applies.
5. "children_left" and "children_right" also are two 1-D arrays. Each index is a proxy Node Id at both. At each index
    the content of these arrays denotes the index of its left or right child's index unless current node is a leaf
            node, in that case it is -1 to indicate left/right child of this node do not exist.

The final "ClusterTree" has been built by traversing the **tree** returned by "DecisionTreeClassifier".
"""

import datetime
import logging
import math
import re
from itertools import product

# from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from Attribute import Attribute
from BasicDataPrep_V1 import calcWeekEndMidEarlyWeek
from ClusterTree import ClusterTreeNode, ClusterTree
from ReadWritePickleFile import readPicklefile

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def is_number_compatible(s):
    """ Returns True if argument is float compatible. """
    try:
        float(s)
        return True
    except ValueError:
        return False


def validationOfClusterConditions(StaticClustering):
    types = StaticClustering['Attributes']['Type']
    types = dict((k.strip().lower(), v) for k, v in types.items())
    ranges = StaticClustering['Attributes']['Ranges']
    ranges = dict((k.strip().lower(), v) for k, v in ranges.items())
    hasOthers = set()
    assert len(set(types.keys()) - set(ranges.keys())) == 0 and \
           len(set(ranges.keys()) - set(types.keys())) == 0, "Attributes mismatch in ranges and types declared."

    for k in types.keys():
        assert (types[k] in ('Categorical', 'Numerical')), \
            "The type of each attribute has to be either 'Numerical' or 'Categorical'"

        if types[k] == 'Numerical':
            minV = ranges[k][0]
            assert (isinstance(minV, int) or isinstance(minV, float)), \
                "The range of a Numerical attribute should be given by 2 numbers!"
            maxV = ranges[k][1]
            assert (isinstance(maxV, int) or isinstance(maxV, float)), \
                "The range of a Numerical attribute should be given by 2 numbers!"
            assert (minV < maxV), \
                "The max value of every numerical attribute should be strickly more than the respective min value"

        elif types[k] == 'Categorical':
            potentialValues = ranges[k]
            assert (isinstance(potentialValues, list) or
                    isinstance(potentialValues, tuple)) and len(potentialValues) > 1, \
                "Please provide the range of every categorical attribute in correct format!"
            potentialValues = [i.lower().strip() if isinstance(i, str) else i for i in potentialValues]
            ranges[k] = potentialValues

    StaticClustering['Attributes']['Type'] = types
    StaticClustering['Attributes']['Ranges'] = ranges

    try:
        others = StaticClustering['Attributes']['Others']
        others = dict((k.strip().lower(), v) for k, v in others.items())
    except KeyError:
        others = None

    categoricalValues = [k for k, v in types.items() if v == 'Categorical']

    for key in StaticClustering['Clusters'].keys():
        cluster = StaticClustering['Clusters'][key]
        newClusterDefination = []

        for criteria in cluster:
            criteria = dict((k.strip().lower(), v) for k, v in criteria.items())
            assert len(set(criteria.keys()) - set(ranges.keys())) == 0, \
                logger.error("Check the Attributes mentioned in of Cluster {}".format(key))

            for k in ranges.keys():

                if types[k] == 'Numerical' and k in criteria.keys():
                    assert (isinstance(criteria[k], list) or isinstance(criteria[k], tuple)) \
                           and is_number_compatible(criteria[k][0]) and is_number_compatible(criteria[k][1]) \
                           and criteria[k][0] >= ranges[k][0] and criteria[k][1] <= ranges[k][1], \
                        logger.error("Check the range of Attribute {}, of Cluster {}".format(k, key))

                elif types[k] == 'Categorical' and k in criteria.keys():
                    assert (isinstance(criteria[k], list) or isinstance(criteria[k], tuple) or
                            isinstance(criteria[k], set)), "Potential values of each categorical \
                    attributes for each cluster definitions must be presented in a list/tuple/set format!"
                    criteria[k] = [i.lower().strip() if isinstance(i, str) else i for i in criteria[k]]

                    if 'others' in criteria[k]:
                        hasOthers.add(k)

                    assert len(set(criteria[k]) - set(ranges[k])) == 0, \
                        logger.error("Check the values of Attribute {}, of Cluster {}".format(k, key))

                else:
                    logger.info("Variable: {} added for cluster: {}, criteria: {}".format(k, key, criteria))
                    criteria[k] = ranges[k]

            newClusterDefination.append(criteria)

        StaticClustering['Clusters'][key] = newClusterDefination

    if len(hasOthers) > 0 and others is None:
        raise ValueError("The definition of 'others' is not present!")
    elif others is not None:
        for k in hasOthers:
            try:
                others[k] = [i.lower().strip() for i in others[k]]
            except KeyError:
                raise ValueError("The definition of 'others' is not defined for attribute {}".format(k))

    return StaticClustering, categoricalValues, others


def DataCuration(StaticClustering):
    types = StaticClustering['Attributes']['Type']
    ranges = StaticClustering['Attributes']['Ranges']
    rKeys = list(ranges.keys())
    curatedData = pd.DataFrame(columns=rKeys + ['cluster'])
    for key in StaticClustering['Clusters'].keys():
        cluster = StaticClustering['Clusters'][key]
        for criteria in cluster:
            data = []
            for k in rKeys:
                if types[k] == 'Numerical' and k in criteria.keys():
                    d = [criteria[k][0], criteria[k][1]]

                elif types[k] == 'Categorical' and k in criteria.keys():
                    d = criteria[k] if isinstance(criteria[k], list) else list(criteria[k])

                elif types[k] == 'Numerical':
                    d = [ranges[k][0], ranges[k][1]]
                elif types[k] == 'Categorical':
                    d = ranges[k] if isinstance(ranges[k], list) else list(ranges[k])
                else:
                    d = ranges[k]

                data.append(d)
            data = [i for i in product(*data)]
            tmp = pd.DataFrame(data, columns=rKeys)
            tmp['cluster'] = key
            curatedData = pd.concat([curatedData, tmp], ignore_index=True)

    return curatedData


def validateNumericDataType(df, categoricalColumns):
    columnsSet = list(df.columns)
    numericCols = [i for i in columnsSet if i not in categoricalColumns + ['cluster']]
    df[numericCols] = df[numericCols].apply(pd.to_numeric)
    return df


def getXandY(df, notNumericCols):
    df = pd.get_dummies(df, columns=notNumericCols)
    xCols = [col for col in df.columns if col != 'cluster']
    X = df[xCols].values
    Y = df['cluster'].values
    return df, X, Y, xCols


def createCART(X, Y):
    # import moved into function to allow inference (which does not use this function) to skip requirement
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(X=X, y=Y)
    assert tree.score(X=X, y=Y) == 1.0, logger.error("Your conditions are NOT mutually exclusive!!!")
    return tree


def getTargetNames(df, tree):
    targets = {}
    for i in df['cluster']:
        if i not in targets.keys():
            targets[i] = 0

        targets[i] += 1

    x = tree.tree_.value[0][0]
    z = sorted(targets.items(), key=lambda x: x[0])
    y = np.array([float(val[1]) for val in z])
    assert np.array_equal(x, y), logger.error(
        "Something went wrong!, Try training with one hot encoded target columns!")
    target_names = [val[0] for val in z]

    return target_names


def getSimpleAttributesCART(data, featureNames, othersGroup):
    assert isinstance(data, pd.DataFrame) and (isinstance(featureNames, list) or
                                               isinstance(featureNames, tuple) or isinstance(featureNames, set))

    minMax = data.describe().loc[['min', 'max']][featureNames]
    uniqueVals = data.nunique()
    attributes = {}
    for att in featureNames:
        attribute = Attribute(att, uniqueVals[att], minMax.loc['max'][att], minMax.loc['min'][att])
        attributes[att] = attribute

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


def buildSimpleClusterTreefromCART(classificationTree, node, depth, simpleTreeNode, featureNames,
                                   target_names, attributes, targetOneHotEncoded=False):
    curNode = ClusterTreeNode()
    if classificationTree.feature[node] == -2:
        curNode.setParent(simpleTreeNode)
        if targetOneHotEncoded:
            targets = [classificationTree.value[node][i][1] for i in range(len(classificationTree.value[node]))]
            result = np.where(targets == max(targets))
            result = result[0][0]
            clusterID = target_names[result]
        else:
            for i in range(len(target_names)):
                if classificationTree.value[node][0][i] > 0:
                    clusterID = target_names[i]
                    break
        clusterID = clusterID.strip().lower()
        curNode.setClusterId(clusterID)
    else:
        i = classificationTree.feature[node]
        attribute = featureNames[i]
        cutValue = classificationTree.threshold[node]

        if attributes[attribute].type == 'Categorical' or attributes[attribute].type == 'Calculated':
            curNode.setAttribute(attribute=attributes[attribute].originalAttribute,
                                 attributeType=attributes[attribute].type)
            curNode.setValue(attributes[attribute].originalAttributeVal)

        else:
            curNode.setAttribute(attribute=attributes[attribute].originalAttribute,
                                 attributeType=attributes[attribute].type)
            curNode.setValue(cutValue)

        curNode.setParent(simpleTreeNode)

        left = buildSimpleClusterTreefromCART(classificationTree, classificationTree.children_left[node],
                                              depth + 1, curNode, featureNames, target_names, attributes,
                                              targetOneHotEncoded)
        curNode.setLeft(left)
        right = buildSimpleClusterTreefromCART(classificationTree, classificationTree.children_right[node],
                                               depth + 1, curNode, featureNames, target_names, attributes,
                                               targetOneHotEncoded)
        curNode.setRight(right)

    return curNode


def getClusterTreeFromRules(staticClusterFilePath, prevVer):
    data = readPicklefile(staticClusterFilePath)
    defaultValues = data['Attributes']['Default']
    defaultValues = dict((k.strip().lower(), v.lower().strip() if isinstance(v, str) else v) for
                         k, v in defaultValues.items())
    data, categoricalColumns, others = validationOfClusterConditions(data)

    df = DataCuration(data)
    newColumnNames = dict()
    for col in df.columns:
        newColumnNames[col] = col.strip().lower()
    df.rename(columns=newColumnNames, inplace=True)
    df, _, _, _, _, _ = calcWeekEndMidEarlyWeek(df)
    df = validateNumericDataType(df, categoricalColumns)
    df, X, Y, xCols = getXandY(df=df, notNumericCols=categoricalColumns)
    tree = createCART(X, Y)
    attributes = getSimpleAttributesCART(df, xCols, others)
    target_names = getTargetNames(df, tree)
    node = buildSimpleClusterTreefromCART(tree.tree_, 0, 1, None, xCols, target_names, attributes, False)
    utc_now = datetime.datetime.now()
    deltaVer = math.ceil(float((utc_now - datetime.datetime(1970, 1, 1)).total_seconds()))
    deltaVer = float("." + str(deltaVer))
    if math.floor(prevVer) == 1:
        curVer = 1.0 + deltaVer
    else:
        curVer = 1.0

    clusterTree = ClusterTree(node, defaultValues, curVer)
    return clusterTree
