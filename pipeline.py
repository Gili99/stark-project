import pandas as pd
import numpy as np
import time

from read_data import read_all_directories
from clusters import Spike, Cluster

from features.FET_time_lag import Time_Lag_Feature
from features.FET_spd import SPD
from features.FET_da import DA
from features.FET_depolarization_graph import DepolarizationGraph
from features.FET_channel_contrast_feature import ChannelContrast
from features.FET_geometrical_estimation import GeometricalEstimation

features = [Time_Lag_Feature(), SPD(), DA(), DepolarizationGraph(), ChannelContrast(), GeometricalEstimation()]


def get_list_of_relevant_waveforms_from_cluster(cluster, spikes_in_waveform = [200]):
    ret = []
    
    for chunk_size in spikes_in_waveform:
        if chunk_size == 0:
            mean = cluster.calc_mean_waveform()
            ret.append([mean])
        elif chunk_size == 1:
            ret.append(cluster.spikes)
        else:
            if cluster.np_spikes is None:
                cluster.finalize_spikes()
            spikes = cluster.np_spikes
            np.random.shuffle(spikes)
            k = spikes.shape[0] // chunk_size
            if k == 0:
                 ret.append([cluster.calc_mean_waveform()])
                 continue
            chunks = np.array_split(spikes, k)
            res = [] 
            for chunk in chunks:
                res.append(Spike(data = chunk.mean(axis = 0)))
            ret.append(res)
        
    return ret

def run():
    clustersGenerator = read_all_directories("dirs.txt")
    headers = []
    for feature in features:
        headers += feature.get_headers()
    headers += ['label']
    
    for clusters in clustersGenerator:
        for cluster in clusters:
            #print('Fixing punits...')
            cluster.fix_punits()
            #print('Dividing data to chunks...')
            chunk_sizes = [0, 100, 200, 500]
            relevantData = get_list_of_relevant_waveforms_from_cluster(cluster, spikes_in_waveform = chunk_sizes)
            for chunk_size, relData in zip(chunk_sizes, relevantData):
                featureMatForCluster = None
                is_first_feature = True
                for feature in features:
                    #print('processing feature ' + feature.name + '...')
                    start_time = time.time()
                    matResult = feature.calculateFeature(relData) # returns a matrix
                    end_time = time.time()
                    #print('processing took %.4f seconds' % (end_time - start_time))
                    if is_first_feature:
                        featureMatForCluster = matResult
                    else:
                        featureMatForCluster = np.concatenate((featureMatForCluster, matResult), axis = 1)

                    is_first_feature = False

                # Append the label for the cluster
                labels = np.ones((len(relData), 1)) * cluster.label
                featureMatForCluster = np.concatenate((featureMatForCluster, labels), axis = 1)
            
                # Save the data to a seperate file (one for each cluster)
                path = "clustersData" + "\\" + str(chunk_size) + '\\' + cluster.get_unique_name() + ".csv"
                df = pd.DataFrame(data = featureMatForCluster)
                df.to_csv(path_or_buf = path, index = False, header = headers)
            print("saved clusters to csv")

if __name__ == "__main__":
    run()


