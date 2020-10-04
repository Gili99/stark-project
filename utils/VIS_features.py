import pandas as pd
import os
import seaborn as sn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import NN_util
import scipy.stats
import numpy as np


DIR = "../clustersData"

def combineFiles():
    df = pd.DataFrame()

    print("Iterating over files...")
    for file in os.listdir(DIR):
        if file.endswith(".csv") and 'all_clusters' not in file:
            path = os.path.join(DIR, file)
            df = df.append(pd.read_csv(path), ignore_index = True) 
    df.to_csv(os.path.join(DIR, 'all_clusters.csv'))

    # drop the label
    df = df.drop(labels = 'label', axis = 1)

    # create and plot the correlation matrix
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot = True)
    plt.show()

def feature_comparison():
    ind = np.arange(13)
    print('Reading data...')
    data = NN_util.read_data(DIR, should_filter = True)
    print('Splitting data...')
    train, _, _ = NN_util.split_data(data, per_train = 1.0, per_dev = 0.0, per_test = 0.0)
    train_squeezed = NN_util.squeeze_clusters(train)
    train_features, train_labels = NN_util.split_features(train_squeezed)
    labels = ['dep_red', 'dep_sd', 'hyp_red', 'hyp_sd', 'fwhm_count' , 'fwhm_sd', 'da', 'da_sd', 'Magnitude_SD', 'magnitude_skewness',
              'graph_avg_speed', 'graph_shortest_distance', 'Channels contrast']
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    pyr_inds = train_labels == 1
    in_inds = train_labels == 0
    pyr_fets = train_features[pyr_inds]
    pyr_means = np.mean(pyr_fets, axis = 0)
    pyr_sem = np.std(pyr_fets, axis = 0)
    in_fets = train_features[in_inds]
    in_means = np.mean(in_fets, axis = 0)
    in_sem = np.std(in_fets, axis = 0)
    width = 0.35

    p1 = plt.bar(ind - width/2, pyr_means, width, yerr = pyr_sem)
    p2 = plt.bar(ind + width/2, in_means, width, yerr = in_sem)
    plt.xticks(ind, labels, rotation = 30, ha = "right", rotation_mode = "anchor")
    plt.show()

if __name__ == "__main__":
    #combineFiles()
    feature_comparison()
