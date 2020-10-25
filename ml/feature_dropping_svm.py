import numpy as np
from gs_svm import grid_search
import time
import ML_util
from itertools import chain, combinations

#indices of the features in the data
indices = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9, 10], [11], [12, 13, 14]]
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
    for comb, acc, pyr_acc, in_acc in results:
        if len(comb) != num_features:
            print('----------------')
            num_features = len(comb)
            print('Results with %d feature(s)' % num_features)
        message = 'Using the features: '
        comb_fets = []
        for ind in comb:
            comb_fets.append(names[ind])
        message += str(comb_fets)
        message += ' general accuracy is: %.3f ;' % acc
        message += 'accuracy on pyr is %.3f ;' % pyr_acc
        message += 'accuracy on in is %.3f' % in_acc
        print(message)
            

def feature_dropping(dataset_path):
    train, dev, test = ML_util.get_dataset(dataset_path)

    combinations = powerset(list(range(len(names))))

    accs = []

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
        _, acc, pyr_acc, in_acc = grid_search(train = train_up, dev = dev_up, test = test_up)
        accs.append((comb, acc, pyr_acc, in_acc))

    print_results(accs)                        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GMM grid search\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created', default = '../data_sets/0_0.60.20.2/')

    args = parser.parse_args()

    dataset_path = args.dataset_path

    feature_dropping(dataset_path)
    
