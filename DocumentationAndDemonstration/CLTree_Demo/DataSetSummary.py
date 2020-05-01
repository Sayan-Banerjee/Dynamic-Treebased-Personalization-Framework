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
        s = s + "Total Number of Data Points: {} ".format(self.getlength())
        s = s + "Attributes: {} ".format(self.getAttributes())
        s = s + "Min Values: {} ".format(self.min_values)
        s = s + "Max Values: {}".format(self.max_values)
        return s


