from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
import pickle
import time

import NN_util
from VIS_model import visualize_model

models = ['svm', 'rf'] #currently we support SVMs and random forests 

def evaluate_predictions(model, clusters, pca):
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
        if pca != None:
           features = pca.transform(features)
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

def run(model = 'svm', save_path = '../saved_models/', load_path = None, n_components = None, use_pca = False, visualize = False, **params):
   if model not in models:
      raise Exception('Model must be in: ' + str(models))
   elif model == 'svm':
      print('Chosen model is SVM')
   elif model == 'rf':
      print('Chosen model is Random Forest')
   print('Reading data...')
   data = NN_util.read_data('../clustersData', should_filter = True)
   print('Splitting data...')
   train, _, test = NN_util.split_data(data, per_train= 0.6, per_dev = 0.2, per_test = 0.2)

   if load_path == None:
      train_squeezed = NN_util.squeeze_clusters(train)
      train_features, train_labels = NN_util.split_features(train_squeezed)

      pca = None

      if use_pca: # if we need to use reduce dimension, we will fit PCA and transform the data
         if n_components == None:
            raise Exception('If use_pca is True but no loading path is given, n_components must be specified')
         if n_components > train[0].shape[1]:
            raise Exception('Number of required components is larger than the number of features')
         pca = PCA(n_components = n_components)
         print('Fitting PCA...')
         pca.fit(train_features)
         print('explained variance by PCA components is: ' + str(pca.explained_variance_ratio_))
         with open(save_path + 'pca', 'wb') as fid: # save PCA model
            pickle.dump(pca, fid)
         print('Transforming training data with PCA...')
         train_features = pca.transform(train_features)

      if model == 'svm':
         clf = svm.SVC(**params)
         print('Fitting SVM model...')
      elif model == 'rf':
         clf = RandomForestClassifier(**params)
         print('Fitting Random Forest model...')
      start = time.time()
      clf.fit(train_features, train_labels)
      end = time.time()
      print('Fitting took %.2f seconds' % (end - start))
      
      with open(save_path + model + '_model', 'wb') as fid: # save the model
         pickle.dump(clf, fid)
   else: # we need to load the model
      print('Loading model...')
      with open(load_path + model + '_model', 'rb') as fid:
         clf = pickle.load(fid)
      if use_pca:
         with open(load_path + 'pca', 'rb') as fid:
            pca = pickle.load(fid)

   print('Evaluating predictions...')
   evaluate_predictions(clf, test, pca)

   if visualize:
      print('Working on visualization...')
      train_squeezed = NN_util.squeeze_clusters(train)
      np.random.shuffle(train_squeezed)
      train_features, train_labels = NN_util.split_features(train_squeezed)
      
      if use_pca:
         train_features = pca.transform(train_features)

      if train_features.shape[1] < 2:
         raise Exception('Cannot visualize data with less than two dimensions')
      
      # this is very costly as we need to predict a grid according to the min and max of the data
      # higher h values woul dresult in a quicker but less accurate graph
      # smaller set might reduce the variability, making the grid smaller
      # it is also possible to specify feature1 and feature2 to pick to wanted dimensions to be showed (default is first two)
      visualize_svm(train_features[:20,:], train_labels[:20], clf, h = 5)  

if __name__ == "__main__":
    #run(load_path = '../saved_models/', use_pca = True)
    #run(model = 'svm', kernel = 'rbf', gamma = 'scale', class_weight = 'balanced')
    run(model = 'rf', n_estimators = 100, class_weight = 'balanced')
    #run(n_components = 2, use_pca = True)
