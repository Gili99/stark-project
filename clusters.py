import numpy as np
import matplotlib.pyplot as plt


class Spike(object):
    def __init__(self, data = None):
        self.data = data # Will contain data from 8 channels each with 32 samples

    def is_punit(self):
        median = np.median(self.data) #an estimate for the baseline as we have depolarization and hyperpolarization, perhaps can be 0, might want it on the avg_spike
        avg_spike = np.mean(self.data, axis = 0) #we look at all channels as one wave, should consider checking each channel separately
        abs_diff = np.absolute(avg_spike - median)
        arg_max = np.argmax(abs_diff, axis = 0) # the axis specification has no effect, just for clarification
        if avg_spike[arg_max] > median:
            return True
        return False

    def fix_punit(self):
        self.data = self.data * -1

    def get_data(self):
        return self.data
        

class Cluster(object):
    def __init__(self):
        self.label = -1
        self.filename = None
        self.numWithinFile = None
        self.shank = None
        self.spikes = []

    def add_spike(self, spike):
        self.spikes.append(spike)

    def get_unique_name(self):
        return self.filename + "_" + str(self.shank) + "_" + str(self.numWithinFile)

    def calc_mean_waveform(self):
        spikes = self.spikes
        mean = np.zeros((8, 32))
        numSpikes = len(spikes)
        for spike in spikes:
            mean += spike.data / numSpikes

        meanSpike = Spike(mean)
        return meanSpike

    def fix_punits(self):
        meanSpike = self.calc_mean_waveform()
        if meanSpike.is_punit():
            self.spikes = [spike.fix_punit() for spike in self.spikes]
        
