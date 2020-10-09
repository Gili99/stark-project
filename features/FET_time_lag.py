import numpy as np

reduction_types = ['ss', 'sa']

class Time_Lag_Feature(object):
    def __init__(self, type_dep = 'ss', type_hyp = 'ss', ratio = 0.3):
        assert type_dep in reduction_types and type_hyp in reduction_types
        
        self.type_dep = type_dep
        self.type_hyp = type_hyp

        self.ratio = ratio

        self.name = 'time lag feature'

    def calculateFeature(self, spikeList):
        result = [self.calc_feature_spike(spike.get_data()) for spike in spikeList]
        result = np.asarray(result)
        return result

    def calc_feature_spike(self, spike):
        # remove channels with lower depolarization than required
        deps = np.min(spike, axis = 1) # max depolarization of each channel
        max_dep = np.min(deps)
        fix_inds = deps >= self.ratio * max_dep 
        spike = spike[fix_inds]

        # find timesteps for depolarizrion in ok chanells and offset according to the main channel
        dep_ind = np.argmin(spike, axis = 1)
        main_chn = np.argmin(spike) // 32 # set main channel to be the one with highest depolariztion
        dep_rel = dep_ind - dep_ind[main_chn] # offsetting

        # calculate sd of depolarization time differences
        dep_sd = np.std(dep_rel)

        # calculate reduction
        if self.type_dep == 'ss':
            dep_red = np.sum(dep_rel ** 2)
        else: #i.e sa
            dep_red = np.sum(np.absolute(dep_rel))

        # find hyperpolarization indeces
        hyp_ind = []
        for i, channel in enumerate(spike):
            trun_channel = channel[dep_ind[i] + 1:]
            hyp_ind.append(trun_channel.argmax() + dep_ind[i] + 1)
        hyp_ind = np.asarray(hyp_ind)

        # repeat calulations                 
        hyp_rel = hyp_ind - hyp_ind[main_chn]
        hyp_sd = np.std(hyp_rel)
        if self.type_hyp == 'ss':
            hyp_red = np.sum(hyp_rel ** 2)
        else: #i.e sa
            hyp_red = np.sum(np.absolute(hyp_rel))
        
        return [dep_red, dep_sd, hyp_red, hyp_sd]

    def get_headers(self):
        return ['dep_red', 'dep_sd', 'hyp_red', 'hyp_sd']

    
