import numpy as np
import matplotlib.pyplot as plt


class Spike(object):
    def __init__(self, data = None):
        self.data = data # Will contain data from 8 channels each with 32 samples

    def is_punit(self):
        median = 0 #np.median(self.data) # an estimate for the baseline as we have depolarization and hyperpolarization, perhaps can be 0, might want it on the avg_spike
        avg_spike = np.mean(self.data, axis = 0) # we look at all channels as one wave, should consider checking each channel separately
        avg_spike = avg_spike[3:-3] # in some cases the hyperpolarization at the edges was stronger than the depolarization, causing wrong conclusion 
        abs_diff = np.absolute(avg_spike - median)
        arg_max = np.argmax(abs_diff, axis = 0) # the axis specification has no effect, just for clarification
        if avg_spike[arg_max] > median:
            return True
        return False

    def fix_punit(self):
        self.data = self.data * -1

    def get_data(self):
        return self.data

    def plot_spike(self):
        num_channels = self.data.shape[0]
        num_timesteps = self.data.shape[1]
        for i in range(num_channels):
            plt.plot([j for j in range(num_timesteps)], self.data[i, :])
        plt.show()

class Cluster(object):
    def __init__(self):
        self.label = -1
        self.filename = None
        self.numWithinFile = None
        self.shank = None
        self.spikes = []
        self.np_spikes = None

    def add_spike(self, spike):
        self.spikes.append(spike)

    def get_unique_name(self):
        return self.filename + "_" + str(self.shank) + "_" + str(self.numWithinFile)

    def calc_mean_waveform(self):
        try:
            if self.np_spikes == None:
                self.finalize_spikes()
        except ValueError: #here because if it actually isn't none there is an error
            pass
        print('Finished finalizing')
        return Spike(data = self.np_spikes.mean(axis = 0))

    def fix_punits(self):
        meanSpike = self.calc_mean_waveform()
        if meanSpike.is_punit():
            print('Punit found, now fixing...')
            self.np_spikes = self.np_spikes * -1

    def finalize_spikes(self):
        print('Started finalizing')
        shape = (len(self.spikes), 8, 32)
        self.np_spikes = np.empty(shape)
        for i, spike in enumerate(self.spikes):
            self.np_spikes[i] = spike.get_data()
        
