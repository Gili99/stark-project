import numpy as np

class ChannelContrast():
    def __init__(self):
        self.name = 'channel contrast feature'

    def find_dominant_channel(self, spike):
        argMinChannel = spike.min(axis=1).argmin()
        argMinTime = spike.min(axis=0).argmin()
        return argMinChannel, argMinTime

    def calculateFeature(self, spikeList):
        result = np.zeros((len(spikeList), 1))
        for i, spike in enumerate(spikeList):

            # Find the dominant channel
            dominantChannel, domTime = self.find_dominant_channel(spike.data)
            reducedArr = spike.data / 100

            # Iterate over the other channels and check the contrast wrt the dominant one
            res = np.zeros((1, 8))
            for j in range(8):
                if j != dominantChannel:
                    dot = np.dot(reducedArr[j], reducedArr[dominantChannel])

                    # if there is a contrast write it, o.w write zero
                    res[0, j] = dot if dot < 0 else 0

            result[i, 0] = np.sum(res * -1)
        return result

    def get_headers(self):
        return ["Channels contrast"]