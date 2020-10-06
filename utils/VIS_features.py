import pandas as pd
import os
import seaborn as sn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

NUM_FETS = 13
DIR = "../clustersData"

def get_df():
    """
    This function reads all files in DIR and creates a pandas data frame from all
    their information, then returns it.
    """
    df = pd.DataFrame()

    print("Iterating over files...")
    for file in os.listdir(DIR):
        if file.endswith(".csv") and 'all_clusters' not in file:
            path = os.path.join(DIR, file)
            df = df.append(pd.read_csv(path), ignore_index = True)

    return df

def corr_matrix():
    """
    The function creates a correlation matrix based on all features in the data
    """
    df = get_df()

    # drop the label
    df = df.drop(labels = 'label', axis = 1)

    # create and plot the correlation matrix
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot = True)
    plt.show()

def feature_comparison():
    """
    The function creates a bar graph comapring all features with separation by cell type
    """
    ind = np.arange(NUM_FETS)
    df = get_df() 

    data = df.to_numpy()
    features, data_labels = data[:,:-1], data[:,-1]
    
    labels = ['dep_red', 'dep_sd', 'hyp_red', 'hyp_sd', 'fwhm_count' , 'fwhm_sd', 'da', 'da_sd', 'Magnitude_SD', 'magnitude_skewness',
              'graph_avg_speed', 'graph_shortest_distance', 'Channels contrast']
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    pyr_inds = data_labels == 1
    in_inds = data_labels == 0
    pyr_fets = features[pyr_inds]
    pyr_means = np.mean(pyr_fets, axis = 0)
    pyr_sem = scipy.stats.sem(pyr_fets, axis = 0)
    in_fets = features[in_inds]
    in_means = np.mean(in_fets, axis = 0)
    in_sem = scipy.stats.sem(in_fets, axis = 0)
    width = 0.35

    p1 = plt.bar(ind - width / 2, pyr_means, width, yerr = pyr_sem)
    p2 = plt.bar(ind + width / 2, in_means, width, yerr = in_sem)
    plt.xticks(ind, labels, rotation = 30, ha = "right", rotation_mode = "anchor")
    plt.legend((p1[0], p2[0]), ('Pyramidal', 'Interneuron'))
    plt.ylabel('Standardized scores')
    plt.show()

def feature_histogram(index):

    df = get_df() 

    data = df.to_numpy()
    features, data_labels = data[:,:-1], data[:,-1]

    pyr_inds = data_labels == 1
    in_inds = data_labels == 0
    pyr_fet = features[pyr_inds][:, index]
    in_fet = features[in_inds][:, index]

    print(len(pyr_fet))
    print(len(pyr_fet[pyr_fet > 10.75]))
    print(len(in_fet))
    print(len(in_fet[in_fet > 10.75]))

    bins = np.arange(0, 11, 0.25)
    print(bins)

    plt.hist(in_fet, bins = bins, alpha = 0.5, label = 'Interneuron', density = True)
    plt.hist(pyr_fet, bins = bins, alpha = 0.5, label = 'Pyramidal', density = True)
    
    plt.legend(loc = 'upper right')
    plt.ylabel('Density')
    plt.xlabel('SD')
    plt.title('Hyperpolarization Standard Deviation Time Lag Density')
    plt.show()

if __name__ == "__main__":
    feature_histogram(3)
