import os
import math
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

def create_batches(data, batch_size, should_shuffle = True):
   """
   This function should recieve the data as np array of waveforms (features and label), and return a list of batches, each batch is a tuple
   of torch's Variable of tensors:
   1) The input features
   2) The corresponding labels
   If should_shuffle (optional, bool, default = True) argument is True we shuffle the data (in place) before creating the batches
   """
   if should_shuffle:
      np.random.shuffle(data)
   batches = []
   cur_batch = 0
   number_of_batches = math.ceil(len(data) / batch_size)
   while cur_batch < number_of_batches:
      batch = data[cur_batch * batch_size: (cur_batch + 1) * batch_size]
      batch = (torch.from_numpy(batch[:,:-1]), torch.from_numpy(batch[:,-1]).long())
      batch = (Variable(batch[0], requires_grad = True), Variable(batch[1]))
      cur_batch += 1
      batches.append(batch)
   return batches

def split_features(data):
   return data[:,:-1], data[:,-1]

def parse_test(data):
   features, label = split_features(data)
   features = Variable(torch.from_numpy(features))
   return features, label
      

def is_legal(cluster):
   """
   This function determines whether or not a cluster's label is legal. It is assumed that all waveforms
   of the cluster have the same label.
   To learn more about the different labels please refer to the pdf file or the read_data.py file
   """
   row = cluster[0]
   return row[-1] >= 0

def read_data(path, should_filter = True):
   """
   The function reads the data from all files in the path.
   It is assumed that each file represeents a single cluster, and have some number of waveforms.
   The should_filter (optional, bool, default = True) argument indicated whether we should filter out
   clusters with problematic label (i.e < 0)
   """
   files = os.listdir(path)
   clusters = []
   for file in files:
      df = pd.read_csv(path+'/'+file)
      nd = df.to_numpy(dtype = 'float32')
      
      if should_filter:
         if is_legal(nd):
            clusters.append(nd)
         else:
            continue
      else:
         clusters.append(nd)
   return np.asarray(clusters)

def split_data(data, per_train = 0.7, per_dev = 0.15, per_test = 0.15 , path = 'data_sets'):
   """
   This function recieves the data as an ndarray. The first level is the different clusters, i.e each file,
   the second level is the different waveforms whithin each clusters and the third is the actual features (with the label)
   The function splits the entire data randomly to train, dev and test sets according to the given precentage.
   It is worth mentioning that although the number of clusters in each set should be according to the function's arguments
   the number of waveforms in each set is actually distributed independently.
   """
   assert per_train + per_dev + per_test == 1
   name = str(per_train) + str(per_dev) + str(per_test) + '/'
   full_path = path + '/' + name if path != None else None
   if path != None and os.path.exists(full_path):
      print('Loading data set from %s...' % (full_path))
      train = np.load(full_path + 'train.npy')
      dev = np.load(full_path + 'dev.npy')
      test = np.load(full_path + 'test.npy')
   else:
      length = data.shape[0]
      print('total number of clusters in data is %d consisting of %d waveforms' % (length, count_waveforms(data)))
   
      per_dev += per_train
   
      np.random.shuffle(data)
      train = data[:math.floor(length * per_train)]
      dev = data[math.floor(length * per_train): math.floor(length * per_dev)]
      test = data[math.floor(length * per_dev):]

      if path != None:
         try:
            os.mkdir(full_path)
         except OSError:
            print ("Creation of the directory %s failed, not saving set" % full_path)
         else:
            print ("Successfully created the directory %s now saving data set" % full_path)
            np.save(full_path + 'train', train)
            np.save(full_path + 'dev', dev)
            np.save(full_path + 'test', test)

   print('total number of clusters in training data is %d consisting of %d waveforms (%.4f%s)' % (train.shape[0], count_waveforms(train), 100 * count_waveforms(train) / count_waveforms(data), '%'))
   print('total number of clusters in dev data is %d consisting of %d waveforms (%.4f%s)' % (dev.shape[0], count_waveforms(dev), 100 * count_waveforms(dev) / count_waveforms(data), '%'))
   print('total number of clusters in test data is %d consisting of %d waveforms (%.4f%s)' % (test.shape[0], count_waveforms(test), 100 * count_waveforms(test) / count_waveforms(data), '%'))
   
   return train, dev, test

def count_waveforms(data):
   """
   This function counts the number of waveforms in all clusters of the data.
   The main usage of this function is statistical data gathering.
   """
   counter = 0
   for cluster in data:
      counter += cluster.shape[0]
   return counter

def squeeze_clusters(data):
   """
   This function receives an nd array with elements with varying sizes.
   It removes the first dimension.
   As numpy doesn't nicely support varying sizes we implement what otherwise could have been achieved using reshape or squeeze 
   """
   res = []
   for cluster in data:
      for waveform in cluster:
         res.append(waveform)
   return np.asarray(res)

