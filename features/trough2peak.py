import numpy as np

class Trough2Peak(object):
    def __init__(self):
        self.name = 'trough2Peak'

    def calculateFeature(self, spikeList):
        result = np.zeros((len(spikeList), 2))

        for i, spike in enumerate(spikeList):
            arr = spike.data
            minChannel = arr.min(axis=1).argmin()
            argMinTime = arr.min(axis=0).argmin()

            arrAfterDep = arr[minChannel, argMinTime:]

            trough = argMinTime
            
            if trough == 31:
                peak = trough
            else:
                peak = arrAfterDep.argmax() + argMinTime

            result[i, 0] = peak - trough

        [np.random.shuffle(spike.data.T) for spike in spikeList]
        for i, spike in enumerate(spikeList):
            arr = spike.data
            minChannel = arr.min(axis=1).argmin()
            argMinTime = arr.min(axis=0).argmin()

            arrAfterDep = arr[minChannel, argMinTime:]

            trough = argMinTime
            
            if trough == 31:
                peak = trough
            else:
                peak = arrAfterDep.argmax() + argMinTime

            result[i, 1] = peak - trough

        return result

    def get_headers(self):
        return ['trough_2_peak', 'trough_2_peak_shuffled']
