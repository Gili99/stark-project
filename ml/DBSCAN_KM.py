from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

import NN_util

models = ['dbscan', 'kmeans'] # currently we support DBSCAN and KMeans 

def evaluate_predictions(model, clusters, pca, scaler):
    total = len(clusters)
    total_pyr = 0
    total_in = 0
    model_clusters = {}
    
    for cluster in clusters:
        features, labels = NN_util.split_features(cluster)
        features = scaler.transform(features)
        if pca != None:
            features = pca.transform(features)
        label = labels[0] # as they are the same for all the cluster
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        preds = model.predict(features)
        for pred in preds:
            if pred not in model_clusters:
                model_clusters[pred] = [label]
            else:
                model_clusters[pred].append(label)
    for model_cluster in model_clusters:
        labels = model_clusters[model_cluster]
        labels = np.asarray(labels)
        total = labels.shape[0]
        counter_pyr = len(labels[labels == 1])
        counter_in = len(labels[labels == 0])
        counter_ut = len(labels[labels < 0])
        print('In cluster %d there are %d examples, of which %d are pyramidal (%.2f), %d are interneurons (%.2f) and %d are untagged (%.2f)' %
              (model_cluster, total, counter_pyr, 100 * counter_pyr / total,  counter_in, 100 * counter_in / total, counter_ut,
               100 * counter_ut / total))

    data = NN_util.squeeze_clusters(clusters)
    features, labels = NN_util.split_features(data)
    features = scaler.transform(features)
    if pca != None:
        features = pca.transform(features)
    x = features[:,8]
    y = features[:,6]
    plt.scatter(x, y, c = model.predict(features))
    plt.show()
    return

def run(model = 'kmeans', save_path = '../saved_models/', load_path = None, n_components = None, use_pca = False, visualize = False, **params):
   if model not in models:
      raise Exception('Model must be in: ' + str(models))
   elif model == 'kmeans':
      print('Chosen model is KMeans')
   elif model == 'dbscan':
      print('Chosen model is DBSCAN')
   print('Reading data...')
   data = NN_util.read_data('../clustersData', should_filter = True)
   print('Splitting data...')
   train, _, _ = NN_util.split_data(data, per_train = 1.0, per_dev = 0.0, per_test = 0.0)

   if load_path == None:
      train_squeezed = NN_util.squeeze_clusters(train)
      train_features, train_labels = NN_util.split_features(train_squeezed)
      scaler = StandardScaler()
      train_features = scaler.fit_transform(train_features)

      pca = None

      if use_pca: # if we need to use reduce dimension, we will fit PCA and transform the data
         if n_components == None:
            raise Exception('If use_pca is True but no loading path is given, n_components must be specified')
         if n_components > train[0].shape[1]:
            raise Exception('Number of required components is larger than the number of features')
         pca = PCA(n_components = n_components, svd_solver = 'full')
         print('Fitting PCA...')
         pca.fit(train_features)
         print('explained variance by PCA components is: ' + str(pca.explained_variance_ratio_))
         with open(save_path + 'pca', 'wb') as fid: # save PCA model
            pickle.dump(pca, fid)
         print('Transforming training data with PCA...')
         #train_features = StandardScaler().fit_transform(train_features)
         train_features = pca.transform(train_features)

      if model == 'kmeans':
         clst = KMeans(**params)
         print('Fitting KMeans model...')
      elif model == 'dbscan':
         clst = DBSCAN(**params)
         print('Fitting DBSCAN model...')
      start = time.time()
      clst.fit(train_features)
      end = time.time()
      print('Fitting took %.2f seconds' % (end - start))
      
      with open(save_path + model + '_model', 'wb') as fid: # save the model
         pickle.dump(clst, fid)
   else: # we need to load the model
      print('Loading model...')
      with open(load_path + model + '_model', 'rb') as fid:
         clst = pickle.load(fid)
      if use_pca:
         with open(load_path + 'pca', 'rb') as fid:
            pca = pickle.load(fid)

   print('Evaluating predictions...')
   evaluate_predictions(clst, train, pca, scaler)

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
      visualize_svm(train_features[:20,:], train_labels[:20], clst, h = 5)  


if __name__ == "__main__":
    run(model = 'kmeans', n_clusters = 2)
    #run(model = 'dbscan', use_pca = True, n_components = 2, eps = 5, min_samples = 2)
