from ClusterTree import ClusterTree

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
            opcode = node.getOperationCode()

            if att not in tree.attributeToDataType.keys():
                if opcode in (2, 3):
                    tree.attributeToDataType[att] = (str,)
                else:
                    tree.attributeToDataType[att] = (int, float)

                attributes.add(att)

            val = node.getDecidingValue()

            if isinstance(val, list) or isinstance(val, tuple) or isinstance(val, set):
                newVals = []
                for v in val:
                    if isinstance(v, int) or isinstance(v, float):
                        v = str(float(v))
                        newVals.append(v)
                    else:
                        if v[0] == '-' and v.replace('-', '', 1).replace('.', '', 1).isdecimal():
                            v = str(float(v))
                        elif v[0] != '-' and v.replace('.', '', 1).isdecimal():
                            v = str(float(v))
                        else:
                            v = v.lower().strip()

                        newVals.append(v)

                node.decidingValue = newVals

            elif opcode == 3:

                if isinstance(val, int) or isinstance(val, float):
                    val = str(float(val))
                else:
                    if val[0] == '-' and val.replace('-', '', 1).replace('.', '', 1).isdecimal():
                        val = str(float(val))
                    elif val[0] != '-' and val.replace('.', '', 1).isdecimal():
                        val = str(float(val))
                    else:
                        val = val.lower().strip()

                node.decidingValue = val

            else:

                val = float(val)
                node.decidingValue = val

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
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
