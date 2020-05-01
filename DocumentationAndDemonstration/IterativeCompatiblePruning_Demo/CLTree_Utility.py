import os
from random import randint
import datetime
import math

def doThisPathExist(path):
    return os.path.exists(path)


def createPath(path):
    if not doThisPathExist(path):
        try:
            os.makedirs(path)
        except RuntimeError:
            return False

        return True


def getRandomStringasId(i=None):
    if i is None:
        utc_now = datetime.datetime.now()
        randomValueAsId = math.ceil(float((utc_now - datetime.datetime(1970, 1, 1)).total_seconds())) + randint(0, 999)
        return str(randomValueAsId)
    else:
        return "DataSet_" + str(i)