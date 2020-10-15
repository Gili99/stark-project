import numpy as np

reduction_types = ['ss', 'sa']

#Direction_Agreeableness
class DA(object):
    def __init__(self, red_type = 'ss'):
        assert red_type in reduction_types
        self.red_type = red_type

        self.name = 'direction agreeableness feature'

    def calculateFeature(self, spikeList):
        result = [self.calc_feature_spike(spike.get_data()) for spike in spikeList]
        result = np.asarray(result)

        copyList = [spike.data.copy() for spike in spikeList]
        [np.random.shuffle(copy.T) for copy in copyList]
        resShuffled = [self.calc_feature_spike(copy) for copy in copyList]
        resShuffled = np.asarray(resShuffled)
        return np.concatenate((result, resShuffled), axis=1)

    def calc_feature_spike(self, spike):
        median = np.median(spike) #might want to work with 0 instead
        direction = spike >= median
        counter = np.sum(direction, axis=0)

        for ind in range(counter.shape[0]):
            temp = counter[ind]
            counter[ind] = temp if temp <= 4 else 8 - temp

        if self.red_type == 'ss':
            res = np.sum(counter ** 2)
        else:
            res = np.sum(counter)

        sd = np.std(counter)

        return [res, sd]

    def get_headers(self):
        return ['da', 'da_sd', 'da_shuffled', 'da_sd_shuffled']
