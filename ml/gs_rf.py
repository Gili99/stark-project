from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import argparse

import ML_util

N = 10

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
        dataset_location = '../data_sets/0' + '_' + str(per_train) + str(per_dev) + str(per_test) + '/'
        train, dev, test = ML_util.get_dataset(dataset_location)

    train_squeezed = ML_util.squeeze_clusters(train)
    dev_squeezed = ML_util.squeeze_clusters(dev)

    train_features, train_labels = ML_util.split_features(train_squeezed)
    dev_features, dev_labels = ML_util.split_features(dev_squeezed)

    n_estimatorss = np.logspace(0, 3, 4).astype('int')
    max_depths = np.logspace(1, 5, 5).astype('int')
    min_samples_splits = np.logspace(1, 5, 5, base = 2).astype('int')
    min_samples_leafs = np.logspace(0, 5, 6, base = 2).astype('int')

    # this is an alternative to the following part using sklearn's grid serach method
    # we use the alternative for better control, mainly running each set of parameters several times
    """
    train_dev = np.concatenate((train_squeezed, dev_squeezed))
    train_dev_features, train_dev_labels = ML_util.split_features(train_dev)
    test_inds = np.concatenate((-1 * np.ones((len(train_squeezed))), np.zeros((len(dev_squeezed)))))
    ps = PredefinedSplit(test_inds)
    
    print()
    parameters = {'n_estimators': n_estimatorss, 'max_depth': max_depths, 'min_samples_split' : min_samples_splits, 'min_samples_leaf' : min_samples_leafs}
    model = RandomForestClassifier(class_weight = 'balanced')
    clf = GridSearchCV(model, parameters, cv = ps)
    print('Starting grid search...')
    start = time.time()
    clf.fit(train_dev_features, train_dev_labels)
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start)) 
    print(clf.best_params_)

    bp = clf.best_params_
    n_estimators, max_depth, min_samples_split, min_samples_leaf = bp['n_estimators'], bp['max_depth'], bp['min_samples_split'], bp['min_samples_leaf']
    classifier = RandomForestClassifier(class_weight = 'balanced', n_estimators = n_estimators, max_depth = max_depth)#, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf) 
    classifier.fit(train_features, train_labels)
    """

    scores = []
    best_score = 0
    params = None
    for n_estimators in n_estimatorss:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    temp_score = 0
                    for i in range(N):
                        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, class_weight = 'balanced')
                        model.fit(train_features, train_labels)
                        score = model.score(dev_features, dev_labels)
                        temp_score += score / N
                    if temp_score > best_score:
                        best_score = temp_score
                        params = (n_estimators, max_depth, min_samples_split, min_samples_leaf)
                        
                    scores.append(temp_score)

    print(params)

    classifier = model = RandomForestClassifier(class_weight = 'balanced', n_estimators = params[0], max_depth = params[1], min_samples_split = params[2], min_samples_leaf = params[3])
    classifier.fit(train_features, train_labels)
    
    print()
    print('Starting evaluation on test set...')
    return evaluate_predictions(classifier, test, verbos)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Random forest grid search\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created', default = '../data_sets/0_0.60.20.2/')
    parser.add_argument('--verbos', type=bool, help='verbosity level (bool)', default = True)
    parser.add_argument('--saving_path', type=str, help='path to save graphs, assumed to be created', default = '../graphs/')
    parser.add_argument('--min_gamma', type=int, help='minimal power of gamma (base 10)', default = -9)
    parser.add_argument('--max_gamma', type=int, help='maximal power of gamma (base 10)', default = -1)
    parser.add_argument('--num_gamma', type=int, help='number of gamma values', default = 36)
    parser.add_argument('--min_c', type=int, help='minimal power of C (base 10)', default = 0)
    parser.add_argument('--max_c', type=int, help='maximal power of C (base 10)', default = 10)
    parser.add_argument('--num_c', type=int, help='number of C values', default = 44)
    parser.add_argument('--kernel', type=int, help='kernael for SVM (notice that different kernels than rbd might require more parameters)', default = 'rbf')


    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    verbos = args.verbos
    saving_path = args.saving_path
    min_gamma = args.min_gamma
    max_gamma = args.max_gamma
    num_gamma = args.num_gamma
    min_c = args.min_c
    max_c = args.max_c
    num_c = args.num_c
    saving_path = args.saving_path
    kernel = args.kernel
    
    grid_search(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma, min_c, max_c, num_c, saving_path, kernel)
    grid_search(verbos = True)
