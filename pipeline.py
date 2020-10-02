import pandas as pd
import numpy as np
import time

from read_data import read_all_directories
from clusters import Spike, Cluster

from FET_time_lag import Time_Lag_Feature
from FET_fwhm import FWHM
from FET_da import DA
from FET_magnitude_distribution import MagnitudeDistribution
from FET_depolarization_graph import DepolarizationGraph
from FET_channel_contrast_feature import ChannelContrast

features = [Time_Lag_Feature(), FWHM(), DA(), MagnitudeDistribution(), DepolarizationGraph(), ChannelContrast()]

data_kind = ['entire', 'hybrid', 'singleton']

def get_list_of_relevant_waveforms_from_cluster(cluster, kind = 'hybrid', spikes_in_waveform = 100):
    assert kind in data_kind

    if kind == 'entire':
        mean = cluster.calc_mean_waveform()
        return [mean]

    if kind == 'singleton':
        return cluster.spikes

    if kind == 'hybrid':
        try:
            if cluster.np_spikes == None:
                cluster.finalize_spikes()
        except ValueError: #here because if it actually isn't none there is an error
            pass
        spikes = cluster.np_spikes
        np.random.shuffle(spikes)
        k = spikes.shape[0] // spikes_in_waveform
        if k == 0:
            return [cluster.calc_mean_waveform()]
        chunks = np.array_split(spikes, k)
        res = [] 
        for chunk in chunks:
            res.append(Spike(data = chunk.mean(axis = 0)))
        return res

def run():
    clustersGenerator = read_all_directories("dirs.txt")
    headers = []
    for feature in features:
        headers += feature.get_headers()
    headers += ['label']
    
    for clusters in clustersGenerator:
        for cluster in clusters:
            print('Fixing punits...')
            cluster.fix_punits()
            print('Dividing data to chunks...')
            relevantData = get_list_of_relevant_waveforms_from_cluster(cluster, kind="entire")
            featureMatForCluster = None
            is_first_feature = True
            for feature in features:
                print('processing feature ' + feature.name + '...')
                start_time = time.time()
                matResult = feature.calculateFeature(relevantData) # returns a matrix
                end_time = time.time()
                print('processing took %.4f seconds' % (end_time - start_time))
                if is_first_feature:
                    featureMatForCluster = matResult
                else:
                    featureMatForCluster = np.concatenate((featureMatForCluster, matResult), axis = 1)

                is_first_feature = False

            # Append the label for the cluster
            labels = np.ones((len(relevantData), 1)) * cluster.label
            featureMatForCluster = np.concatenate((featureMatForCluster, labels), axis = 1)
            
            # Save the data to a seperate file (one for each cluster)
            path = "clustersData" + "\\" + cluster.get_unique_name() + ".csv"
            df = pd.DataFrame(data=featureMatForCluster)
            df.to_csv(path_or_buf=path, index=False, header = headers)
        print("saved clusters to csv")

if __name__ == "__main__":
    run()


