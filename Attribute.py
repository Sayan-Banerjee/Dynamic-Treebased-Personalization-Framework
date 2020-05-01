import re
import logging

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Attribute:
    def __init__(self, name, noOfUniqueValues, maxVal, minVal):
        self.name = name
        self.maxVal = maxVal
        self.minVal = minVal
        self.noOfUniqueValues = noOfUniqueValues
        self.setType()
        self.setOriginalAttribute()

    def setType(self):
        self.type = 'Categorical' if self.noOfUniqueValues == 2 and self.maxVal == 1. and self.minVal == 0. else 'Numerical'
        if self.type == 'Categorical' and re.findall("(_others|_weekend|_midweek|_earlyweek)$",
                                                     self.name):
            self.type = 'Calculated'

    def setOriginalAttribute(self):
        if self.type == 'Calculated':
            x = re.findall("(_others|_weekend|_midweek|_earlyweek)$", self.name)
            if x:
                x = re.search("(_others|_weekend|_midweek|_earlyweek)$", self.name)
                self.originalAttribute = self.name[:x.start()].strip().lower()
                self.originalAttributeVal = None
            else:
                self.originalAttribute = self.name.strip().lower()
                self.originalAttributeVal = None

        elif self.type == 'Categorical':
            x = self.name.find("_")
            # x = re.findall("_[a-z|A-Z|0-9]+$", self.name)
            if x > -1:
                # x = re.search("_[a-z|A-Z|0-9]+$", self.name)
                # self.originalAttribute = self.name[:x.start()].strip().lower()
                self.originalAttribute = self.name[:x].strip().lower()
                # self.originalAttributeVal = int(self.name[x.start()+1:]) if self.name[x.start()+1:].isdecimal()\
                # else self.name[x.start()+1:].lower()
                val = self.name[x + 1:]
                self.originalAttributeVal = float(val) if val.isdecimal() else val.strip().lower()

                if isinstance(self.originalAttributeVal, str):
                    if self.originalAttributeVal[0] == '-' and \
                            self.originalAttributeVal.replace('-', '', 1).replace('.', '', 1).isdecimal():

                        self.originalAttributeVal = float(self.originalAttributeVal)

                    elif self.originalAttributeVal.replace('.', '', 1).isdecimal():

                        self.originalAttributeVal = float(self.originalAttributeVal)
            else:
                self.originalAttribute = self.name.strip().lower()
                self.originalAttributeVal = None
        else:
            self.originalAttribute = self.name.strip().lower()
            self.originalAttributeVal = None

    def setOriginalAttributeVal(self, val):
        if self.type == 'Calculated' and (isinstance(val, list) or isinstance(val, tuple)):
            self.originalAttributeVal = val
        else:
            logger.error("You are not allowed to set the value!")
