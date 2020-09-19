import pandas as pd
import numpy as np
from read_data import read_all_directories
from clusters import Spike, Cluster

from feature1 import Feature1

features = [Feature1()]

def calc_mean_waveform(spikes):
    mean = np.zeros((8, 32))
    numSpikes = len(spikes)
    for spike in spikes:
        mean += spike.data / numSpikes

    meanSpike = Spike(mean)
    return meanSpike

def get_list_of_relevant_waveforms_from_cluster(cluster, kind = 'hybrid', spikes_in_waveform = 100):
    assert kind == 'entire' or kind == 'hybrid' or kind == 'singelton'

    if kind == 'entire':
        mean = calc_mean_waveform(cluster.spikes)
        return [mean]

    if kind == 'singelton':
        return cluster.spikes

    if kind == 'hybrid':
        spikes = np.asarray(cluster.spikes)
        np.random.shuffle(spikes)
        k = spikes.shape[0]//spikes_in_waveform
        if k == 0:
            return [calc_mean_waveform(cluster.spikes)]
        waveforms = 0
        res = [] 
        while waveforms < k:
            res.append(calc_mean_waveform(spikes[waveforms * spikes_in_waveform: (waveforms + 1) * spikes_in_waveform]))
            waveforms += 1
        return res
        
"""
def calc_length_of_features():
    num = 0
    for feature in features:
        num += feature.get_num_of_returned_features()
    return num"""

def run():
    clusters = read_all_directories("dirs.txt")
    numClusters = len(clusters)
    for cluster in clusters.values():
        relevantData = get_list_of_relevant_waveforms_from_cluster(cluster)
        featureMatForCluster = None
        for feature in features:
            matResult = feature.calculateFeature(relevantData) # returns a matrix
            
            if featureMatForCluster == None:
                featureMatForCluster = matResult
            else:
                featureMatForCluster = np.concatenate((featureMatForCluster, matResult), axis=1)

        # Append the label for the cluster
        labels = np.ones((len(relevantData), 1)) * cluster.label
        featureMatForCluster = np.concatenate((featureMatForCluster, labels), axis=1)
        
        # Save the data to a seperate file (one for each cluster)
        path = "clustersData" + "\\" + cluster.get_unique_name() + ".csv"
        df = pd.DataFrame(data=featureMatForCluster)
        df.to_csv(path_or_buf=path, index=False, header=["a", "b", "c", "label"])

if __name__ == "__main__":
    run()


