from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

import ML_util

def evaluate_predictions(model, clusters, verbos = False):
    total = len(clusters)
    total_pyr = 0
    total_in = 0
    correct_pyr = 0
    correct_in = 0
    correct_chunks = 0
    total_chunks = 0
    correct_clusters = 0
    for cluster in clusters:
        features, labels = ML_util.split_features(cluster)
        label = labels[0] #as they are the same for all the cluster
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        preds = model.predict(features)
        prediction = round(preds.mean())
        total_chunks += preds.shape[0]
        correct_chunks += preds[preds == label].shape[0]
        correct_clusters += 1 if prediction == label else 0
        correct_pyr += 1 if prediction == label and label == 1 else 0
        correct_in += 1 if prediction == label and label == 0 else 0

    if verbos:
        print('Number of correct classified clusters is %d, which is %.4f%s' % (correct_clusters, 100 * correct_clusters / total, '%'))
        print('Number of correct classified chunks is %d, which is %.4f%s' % (correct_chunks, 100 * correct_chunks / total_chunks, '%'))
        print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
        pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
        in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in
        print('%.4f%s of pyrmidal cells classified correctly' % (pyr_percent, '%'))
        print('%.4f%s of interneurons classified correctly' % (in_percent, '%'))
    return correct_clusters, correct_clusters / total


def grid_search(verbos = False, train = None, dev = None, test = None):
    if train is None or dev is None or test is None:
        per_train = 0.6
        per_dev = 0.2
        per_test = 0.2
        ML_util.create_datasets(per_train, per_dev, per_test)
        dataset_location = '../data_sets/clustersData_hybrid_200' + '_' + str(per_train) + str(per_dev) + str(per_test) + '/'
        train, dev, test = ML_util.get_dataset(dataset_location)

    train_squeezed = ML_util.squeeze_clusters(train)
    dev_squeezed = ML_util.squeeze_clusters(dev)

    train_dev = np.concatenate((train_squeezed, dev_squeezed))
    train_dev_features, train_dev_labels = ML_util.split_features(train_dev)
    test_inds = np.concatenate((-1 * np.ones((len(train_squeezed))), np.zeros((len(dev_squeezed)))))
    ps = PredefinedSplit(test_inds)         

    #n_estimatorss = np.logspace(0, 4, 5).astype('int')
    #max_depths = np.logspace(1, 3, 6).astype('int') #remember to check none as well
    #min_samples_splits = np.logspace(1, 7, 14, base = 2).astype('int')
    min_samples_leafs = np.logspace(1, 5, 10, base = 2).astype('int')
    
    print()
    parameters = {'min_samples_leaf': min_samples_leafs}#, 'max_depth': max_depths, 'min_samples_split' : min_samples_splits, 'min_samples_leaf' : min_samples_leafs}
    model = RandomForestClassifier(class_weight = 'balanced', n_estimators = 100, max_depth = 25, min_samples_split = 9)
    clf = GridSearchCV(model, parameters, cv = ps)
    print('Starting grid search...')
    start = time.time()
    clf.fit(train_dev_features, train_dev_labels)
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start)) 
    print(clf.best_params_)

    print()
    print('Starting evaluation on test set...')
    return evaluate_predictions(clf, test, verbos)

if __name__ == "__main__":
    grid_search(verbos = True)
