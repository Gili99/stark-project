from sklearn import svm
import numpy as np
import pickle
import time

import NN_util

def split_features(data):
   return data[:,:-1], data[:,-1] 

def evaluate_predictions(model, clusters):
    total = len(clusters)
    total_pyr = 0
    total_in = 0
    correct_pyr = 0
    correct_in = 0
    #correct_waveforms = 0
    correct_clusters = 0
    for cluster in clusters:
        features, labels = split_features(cluster)
        label = labels[0] #as they are the same for all the cluster
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        preds = model.predict(features)
        prediction = round(pred.mean())
        correct_clusters += 1 if prediction == label else 0
        correct_pyr += 1 if prediction == label and label == 1 else 0
        correct_in += 1 if prediction == label and label == 0 else 0

    print('Number of correct classified clusters is %d, which is %.4f%s' % (correct_clusters, 100 * correct_clusters / total, '%'))
    print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
    pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
    in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in
    print('%.4f%s of pyrmidal cells classified correctly' % (pyr_percent, '%'))
    print('%.4f%s of interneurons classified correctly' % (in_percent, '%'))
    return correct_clusters, correct_clusters / total

def run():
    print('Reading data...')
    data = NN_util.read_data('clustersData', should_filter = True)
    print('Splitting data...')
    train, _, test = NN_util.split_data(data, per_train= 0.6, per_dev = 0.0, per_test = 0.4)
    train = NN_util.squeeze_clusters(train)
    train_features, train_labels = split_features(train)

    clf = svm.SVC(kernel='rbf', gamma = 'auto')
    print('Fitting SVM model...')
    start = time.time()
    clf.fit(train_features, train_labels)
    end = time.time()
    print('Fitting took %.2f seconds' % (end - start))

    print('Evaluating predictions...')
    evaluate_predictions(clf, test)

    #TODO - save model (maybe allow loading model to)
    #TODO - handle bias in data
    #TODO - visualize? will require using two dimnesions, perhaps run pca beforhand if wanted

if __name__ == "__main__":
    run()
