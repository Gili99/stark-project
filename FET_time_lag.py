import numpy as np

reduction_types = ['ss', 'sa']

class Time_Lag_Feature(object):
    def __init__(self, type_dep = 'ss', type_hyp = 'ss'):
        assert type_dep in reduction_types and type_hyp in reduction_types
        
        self.type_dep = type_dep
        self.type_hyp = type_hyp

        self.name = 'time lag feature'

    def calculateFeature(self, spikeList):
        result = [self.calc_feature_spike(spike.get_data()) for spike in spikeList]
        result = np.asarray(result)
        return result

    def calc_feature_spike(self, spike):
        dep_ind = np.argmin(spike, axis = 1)
        main_chn = np.argmin(spike) // 32
        dep_rel = dep_ind - dep_ind[main_chn]
        dep_sd = np.std(dep_rel)
        if self.type_dep == 'ss':
            dep_red = np.sum(dep_rel ** 2)
        else: #i.e sa
            dep_red = np.sum(np.absolute(dep_rel))

        first_dep = np.min(dep_ind)
        trun_spike = spike.T[first_dep + 1:].T
        hyp_ind = np.argmax(trun_spike, axis = 1) + first_dep + 1
        hyp_rel = hyp_ind - hyp_ind[main_chn]
        hyp_sd = np.std(hyp_rel)
        if self.type_hyp == 'ss':
            hyp_red = np.sum(hyp_rel ** 2)
        else: #i.e sa
            hyp_red = np.sum(np.absolute(hyp_rel))
        
        return [dep_red, dep_sd, hyp_red, hyp_sd]

    def get_headers(self):
        return ['dep_red', 'dep_sd', 'hyp_red', 'hyp_sd']

    
