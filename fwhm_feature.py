import numpy as np

class FWHM(object):
    def __init__(self):
        pass

    def calculateFeature(self, spikeList):
        result = [self.calc_feature_spike(spike.get_data()) for spike in spikeList]
        result = np.asarray(result)
        return result

    def calc_feature_spike(self, spike):
        dep = np.min(spike, axis = 1)
        main_chn = np.argmin(spike) // 32
        rel_dep = dep / dep[main_chn]
        count = np.count_nonzero(rel_dep > 0.5)
        sd = np.std(rel_dep)
        return [count, sd]

    def get_headers(self):
        return ['fwhm_count', 'fwhm_sd']
