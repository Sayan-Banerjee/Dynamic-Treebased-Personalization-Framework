import os
import shutil
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


def clearTheDirectory(path):
    if not doThisPathExist(path):
        raise FileNotFoundError("Error! The passed path do not exist!")
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            except Exception as e:
                raise RuntimeError('Failed to delete %s. Reason: %s' % (file_path, e))

        return True
