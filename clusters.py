
class Spike(object):
    def __init__(self, data = None):
        self.data = data # Will contain data from 8 channels each with 32 samples

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