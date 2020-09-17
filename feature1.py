import abc
import numpy as np

class Feature1(object):
    def __init__(self):
        pass

    def calculateFeature(self, spikeList):
        result = np.zeros((len(spikeList), 3))
        return result