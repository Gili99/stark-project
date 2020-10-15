import numpy as np

class SPD(object):
    def __init__(self, ratio = 0.5):
        self.ratio = ratio

        self.name = 'spatial dispersion feature'

    def calculateFeature(self, spikeList):
        result = [self.calc_feature_spike(spike.get_data()) for spike in spikeList]
        result = np.asarray(result)

        [np.random.shuffle(spike.data.T) for spike in spikeList]
        resShuffled = [self.calc_feature_spike(spike.get_data()) for spike in spikeList]
        resShuffled = np.asarray(resShuffled)
        return np.concatenate((result, resShuffled), axis=1)

    def calc_feature_spike(self, spike):
        dep = np.min(spike, axis = 1)
        main_chn = np.argmin(spike) // 32
        rel_dep = dep / dep[main_chn]
        count = np.count_nonzero(rel_dep > self.ratio)
        sd = np.std(rel_dep)
        return [count, sd]

    def get_headers(self):
        return ['spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_count_shuffled', 'spatial_dispersion_sd_shuffled']
