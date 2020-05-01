from ClusterTree import *

def getAttributesAndClusters(tree):
    assert isinstance(tree, ClusterTree)
    
    queue = Queue()
    root = tree.getRoot()
    attributes = set()
    clusters = set()

    queue.enqueue(root)
    while not queue.isEmpty(): 
        node = queue.dequeue()
        if not node.isLeaf():
            att = node.getAttribute()
            if not att in tree.attributeToDataType.keys():
                val = node.getDecidingValue()
                if isinstance(val, list) or isinstance(val, tuple):
                    val = val[0]
                if isinstance(val, str):
                    tree.attributeToDataType[att] = (str,)
                else:
                    tree.attributeToDataType[att] = (int, float)
                    
            attributes.add(att)
            
            lChild = node.getLeft()
            queue.enqueue(lChild)
            rChild = node.getRight()
            queue.enqueue(rChild)
        else:
            cluster = node.getClusterId()
            clusters.add(cluster)


    return attributes, clusters

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

