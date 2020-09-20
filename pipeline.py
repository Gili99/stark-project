import pandas as pd
import numpy as np

from read_data import read_all_directories
from clusters import Spike, Cluster

from time_lag_feature import Time_Lag_Feature
from fwhm_feature import FWHM
from da_feature import DA

features = [Time_Lag_Feature(), FWHM(), DA()]

data_kind = ['entire', 'hybrid', 'singleton']

def calc_mean_waveform(spikes):
    mean = np.zeros((8, 32))
    numSpikes = len(spikes)
    for spike in spikes:
        mean += spike.get_data() / numSpikes

    meanSpike = Spike(mean)
    return meanSpike

def get_list_of_relevant_waveforms_from_cluster(cluster, kind = 'hybrid', spikes_in_waveform = 100):
    assert kind in data_kind

    if kind == 'entire':
        mean = calc_mean_waveform(cluster.spikes)
        return [mean]

    if kind == 'singleton':
        return cluster.spikes

    if kind == 'hybrid':
        spikes = np.asarray(cluster.spikes)
        np.random.shuffle(spikes)
        k = spikes.shape[0]//spikes_in_waveform
        if k == 0:
            return [cluster.calc_mean_waveform()]
        waveforms = 0
        res = [] 
        while waveforms < k:
            res.append(calc_mean_waveform(spikes[waveforms * spikes_in_waveform: (waveforms + 1) * spikes_in_waveform]))
            waveforms += 1
        return res

def run():
    clusters = read_all_directories("dirs.txt")
    numClusters = len(clusters)
    headers = []
    for feature in features:
        headers += feature.get_headers()
    headers += ['label']
    
    for cluster in clusters.values():
        cluster.fix_punits()
        relevantData = get_list_of_relevant_waveforms_from_cluster(cluster)
        featureMatForCluster = None
        is_first_feature = True
        for feature in features:
            matResult = feature.calculateFeature(relevantData) # returns a matrix
            
            if is_first_feature:
                featureMatForCluster = matResult
            else:
                featureMatForCluster = np.concatenate((featureMatForCluster, matResult), axis=1)

            is_first_feature = False

        # Append the label for the cluster
        labels = np.ones((len(relevantData), 1)) * cluster.label
        featureMatForCluster = np.concatenate((featureMatForCluster, labels), axis=1)
        
        # Save the data to a seperate file (one for each cluster)
        path = "clustersData" + "\\" + cluster.get_unique_name() + ".csv"
        df = pd.DataFrame(data=featureMatForCluster)
        df.to_csv(path_or_buf=path, index=False, header = headers)

if __name__ == "__main__":
    run()


