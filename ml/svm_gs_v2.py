from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import time

import NN_util
from VIS_heatmap import create_heatmap

def evaluate_predictions(model, clusters):
    total = len(clusters)
    total_pyr = 0
    total_in = 0
    correct_pyr = 0
    correct_in = 0
    correct_chunks = 0
    total_chunks = 0
    correct_clusters = 0
    for cluster in clusters:
        features, labels = NN_util.split_features(cluster)
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

    print('Number of correct classified clusters is %d, which is %.4f%s' % (correct_clusters, 100 * correct_clusters / total, '%'))
    print('Number of correct classified chunks is %d, which is %.4f%s' % (correct_chunks, 100 * correct_chunks / total_chunks, '%'))
    print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
    pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
    in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in
    print('%.4f%s of pyrmidal cells classified correctly' % (pyr_percent, '%'))
    print('%.4f%s of interneurons classified correctly' % (in_percent, '%'))
    return correct_clusters, correct_clusters / total


def run():
    per_train = 0.6
    per_dev = 0.2
    per_test = 0.2
    NN_util.create_datasets(per_train, per_dev, per_test)
    dataset_location = '../data_sets/clustersData_default' + '_' + str(per_train) + str(per_dev) + str(per_test) + '/'
    train, dev, test = NN_util.get_dataset(dataset_location)

    train_squeezed = NN_util.squeeze_clusters(train)
    dev_squeezed = NN_util.squeeze_clusters(dev)

    train_dev = np.concatenate((train_squeezed, dev_squeezed))
    train_dev_features, train_dev_labels = NN_util.split_features(train_dev)
    test_inds = np.concatenate((-1 * np.ones((len(train_squeezed))), np.zeros((len(dev_squeezed)))))
    ps = PredefinedSplit(test_inds)         

    print()
    parameters = {'C': np.logspace(1, 6, 12), 'gamma': np.logspace(-9, -2, 16)}
    model = svm.SVC(kernel = 'rbf', class_weight = 'balanced')
    clf = GridSearchCV(model, parameters, cv = ps)
    print('Starting grid search...')
    start = time.time()
    clf.fit(train_dev_features, train_dev_labels)
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start)) 
    print(clf.best_params_)

    scores = clf.cv_results_['mean_test_score']
    cs = [round(v, 3) for v in np.logspace(1, 6, 12)]
    gammas = [round(v, 9) for v in np.logspace(-9, -2, 16)]
    create_heatmap(gammas, cs, 'Gamma', 'C', 'SVM Grid Search', scores.reshape((len(cs), len(gammas))))

    print()
    print('Starting evaluation on test set...')
    evaluate_predictions(clf, test)

if __name__ == "__main__":
    run()
