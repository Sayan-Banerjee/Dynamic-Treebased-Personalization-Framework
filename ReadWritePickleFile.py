import logging

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def readPicklefile(path):
    try:
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        return data
    except IOError:
        logger.error("Could not read the file at {}".format(path))
        raise IOError("Could not read the file at {}".format(path))


def writePickleFile(path, data):
    try:
        with open(path, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            return True
    except IOError:
        logger.error("Could not write to the file at {}".format(path))
        raise IOError("Could not write to the file at {}".format(path))
