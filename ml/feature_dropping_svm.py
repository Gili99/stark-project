import numpy as np
from gs_svm import grid_search
import time
import NN_util
from itertools import chain, combinations

#indices of the features in the data
indices = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9, 10], [11], [12, 13]]
#name of each feature, corresponding to the indices list
names = ['time lag', 'spatial dispersion', 'direction agreeableness', 'graph speeds', 'channels contrast', 'geometrical']

def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

def remove_features(keep, data):
    clusters = []
    for cluster in data:
        clusters.append(cluster[:, keep])
    return np.asarray(clusters)

def print_results(results):
    num_features = 0
    for comb, acc in results:
        if len(comb) != num_features:
            print('----------------')
            num_features = len(comb)
            print('Results with %d feature(s)' % num_features)
        message = 'Using the features: '
        comb_fets = []
        for ind in comb:
            comb_fets.append(names[ind])
        message += str(comb_fets)
        message += ' accuracy is: '
        print(message + '%.3f' % acc)
            

def feature_dropping():
    per_train = 0.6
    per_dev = 0.2
    per_test = 0.2
    NN_util.create_datasets(per_train, per_dev, per_test)
    dataset_location = '../data_sets/clustersData_default' + '_' + str(per_train) + str(per_dev) + str(per_test) + '/'
    train, dev, test = NN_util.get_dataset(dataset_location)

    combinations = powerset(list(range(len(names))))

    results = []

    for comb in combinations:
        inds = []
        comb_fets = []
        message = 'Used features are '
        for i in comb:
            inds += indices[i]
            comb_fets.append(names[i])
        message += str(comb_fets)
        print(message)
        inds.append(-1)
        train_up = remove_features(inds, train)
        dev_up = remove_features(inds, dev)
        test_up = remove_features(inds, test)
        _, accuracy = grid_search(train = train_up, dev = dev_up, test = test_up)
        results.append((comb, accuracy))

    print_results(results)                        

if __name__ == "__main__":
    feature_dropping()
    
