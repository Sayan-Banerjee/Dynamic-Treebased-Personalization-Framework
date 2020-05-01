import logging

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ClusterTreeNode:
    def __init__(self):
        self.operationCode = None
        self.attribute = None
        self.decidingValue = None
        self.clusterID = 'DEFAULT'
        self.left = None
        self.right = None
        self.parent = None
        self.fractionInherited=0.
        self.attributeType = None

    def setAttribute(self, attribute, attributeType):
        self.attribute = attribute
        self._SetAttributeType(attributeType)

    def _SetAttributeType(self, attributeType):
        self.attributeType = attributeType

    def getAttributeType(self):
        return self.attributeType

    def getAttribute(self):
        return self.attribute

    def setValue(self, value):
        self.decidingValue = value
        self._setOperationCode(value)

    def getDecidingValue(self):
        return self.decidingValue

    def _setOperationCode(self, value):
        if isinstance(value, list) or isinstance(value, tuple):
            self.operationCode= 2 # Operation code stands for not in and in
        elif self.attributeType=='Categorical': # Operation Code Stands for != and ==
            self.operationCode= 3
        else:
            self.operationCode= 1 # operatopm code stand for <= and >

    def getOperationCode(self):
        return self.operationCode

    def isLeaf(self):
        if self.left==None and self.right ==None:
            return True
        else:
            return False

    def setRight(self, node):
        self.right = node

    def getRight(self):
        return self.right

    def getLeft(self):
        return self.left

    def setLeft(self, node):
        self.left = node

    def setParent(self, node):
        self.parent = node

    def getParent(self):
        return self.parent

    def setInheriatedFraction(self, value):
        if value>=0.0 and value<=1.0:
            self.fractionInherited=value
        else:
            logger.error("The fraction should be greater than 0 and less than 1!")
    def getInheriatedFraction(self):
        return self.fractionInherited

    def setClusterId(self, clusterId):
        self.clusterID = clusterId

    def getClusterId(self):
        return self.clusterID

class ClusterTree:
    def __init__(self, node, defaultVals, versionOfClusterAlgo):
        self.setRoot(node)
        self.defaultVals = dict((k.lower().strip().lower(), v) for k,v in defaultVals.items())
        self.versionOfClusterAlgo = versionOfClusterAlgo
        self.attributeToDataType = dict()

    def setRoot(self, node):
        assert isinstance(node, ClusterTreeNode) and node.parent is None
        self.root = node
        return

    def isRoot(self, node):
        if self.root is node:
            return True
        else:
            return False
    def getRoot(self):
        return self.root

    def getClusterID(self, info):
        ## Iterative
        assert isinstance(info, dict), logger.error("ClusterTree.getClusterID() expects information to be passed in\
        python dictionary(key-value pair) format!")

        info = dict((k.lower().strip().lower(), v) for k,v in info.items())

        curNode = self.root
        clusterID = None
        while clusterID is None and curNode is not None:
            if curNode.isLeaf():
                clusterID = curNode.getClusterId()
                break

            val = curNode.getDecidingValue()
            if isinstance(val, str):
                val = val.strip().lower()

            opcode = curNode.getOperationCode()
            att = curNode.getAttribute()

            nodeVal = info.get(att, self.defaultVals[att])
            if isinstance(nodeVal, str):
                nodeVal = nodeVal.strip().lower()
            else:
                nodeVal = str(nodeVal)
            acceptedDtypes = self.attributeToDataType[att]
            flag = False
            for typ in acceptedDtypes:
                flag = flag or isinstance(nodeVal, typ)

            if not flag:
                if isinstance(nodeVal, str) and nodeVal.replace('.','',1).isdecimal():
                    nodeVal = float(nodeVal)

            if opcode == 1:
                if isinstance(nodeVal, str):
                    logger.error("Unexpected data for Attribute: {}, Passed Value: {}, Numerical data expected!".format(att, nodeVal))
                    nodeVal = self.defaultVals[att]
                if nodeVal<= val:
                    curNode = curNode.getLeft()
                else:
                    curNode = curNode.getRight()
            elif opcode == 2:
                if nodeVal not in val:
                    curNode = curNode.getLeft()
                else:
                    curNode = curNode.getRight()
            elif opcode == 3:
                if nodeVal != val:
                    curNode = curNode.getLeft()
                else:
                    curNode = curNode.getRight()

        return clusterID, self.versionOfClusterAlgo
