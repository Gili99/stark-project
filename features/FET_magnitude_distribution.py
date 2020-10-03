import numpy as np

class MagnitudeDistribution(object):
    def __init__(self):
        self.name = 'magnitude distribution feature'

    def calculateMean(self, dist):
        mean = 0
        for i, p in enumerate(dist):
            mean += i * p
        return mean

    def calculateSD(self, dist):
        mean = self.calculateMean(dist)
        sd = 0
        for i, p in enumerate(dist):
            sd += ((i - mean)** 2) * p
        return sd

    def calculateSkewness(self, dist):
        mean = self.calculateMean(dist)
        sd = self.calculateSD(dist)
        skew = 0
        for i, p in enumerate(dist):
            skew += ((i - mean) / sd) ** 3
        return skew

    def calculateFeature(self, spikeList):
        result = np.zeros((len(spikeList), 2))
        absolute = lambda x: abs(x)
        vfunc = np.vectorize(absolute)

        for i, spike in enumerate(spikeList):
            arrSum = [0 for i in range(32)]
        
            # Transfer the spike into absolute values and sum these values for each timestamp
            absArr = vfunc(spike.data)
            for j in range(32):
                arrSum[j] = sum(absArr[:, j])

            # turn the histogram into a distribution
            totalMagnitude = sum(arrSum)
            dist = list(map(lambda x: x/totalMagnitude, arrSum))

            # Calculate SD and skewness and put it in the result
            result[i, 0] = self.calculateSD(dist)
            result[i, 1] = self.calculateSkewness(dist)

        return result

    def get_headers(self):
        return ["Magnitude_SD", "magnitude_skewness"]
