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
   return features, label[0]
      

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
   clusters with problematic label (i.e. < 0)
   """
   files = os.listdir(path)
   clusters = []
   for file in sorted(files):
      df = pd.read_csv(path + '/' + file)
      nd = df.to_numpy(dtype = 'float32')
      
      if should_filter:
         if is_legal(nd):
            clusters.append(nd)
         else:
            continue
      else:
         clusters.append(nd)
   return np.asarray(clusters)

def create_datasets(per_train = 0.6, per_dev = 0.2, per_test = 0.2, datasets = 'datas.txt', should_filter = True, save_path = '../data_sets'):
   paths = []
   with open(datasets, 'r') as fid:
      while True:
         path = fid.readline()
         if path == '':
            break
         else:
            paths.append(path.rstrip())
   names = [path.split('/')[-1] + '_' for path in paths]
   
   inds = []
   inds_initialized = False
   for name, path in zip(names, paths):
      print('Reading data from %s...' % path)
      data = read_data(path, should_filter)
      if not inds_initialized:
         inds = np.arange(data.shape[0])
         np.random.shuffle(inds)
         inds_initialized = True
      print('Splitting %s set...' % name)
      split_data(data, per_train = per_train, per_dev = per_dev, per_test = per_test, inds = inds, path = save_path, data_name = name)

def get_dataset(path):
   print('Loading data set from %s...' % (path))
   train = np.load(path + 'train.npy')
   dev = np.load(path + 'dev.npy')
   test = np.load(path + 'test.npy')

   return train, dev, test
   
def split_data(data, per_train = 0.6, per_dev = 0.2, per_test = 0.2 , path = '../data_sets', should_load = True, data_name = '', inds = []):
   """
   This function recieves the data as an ndarray. The first level is the different clusters, i.e each file,
   the second level is the different waveforms whithin each clusters and the third is the actual features (with the label)
   The function splits the entire data randomly to train, dev and test sets according to the given precentage.
   It is worth mentioning that although the number of clusters in each set should be according to the function's arguments
   the number of waveforms in each set is actually distributed independently.
   """
   assert per_train + per_dev + per_test == 1
   name = data_name + str(per_train) + str(per_dev) + str(per_test) + '/'
   full_path = path + '/' + name if path != None else None
   if path != None and os.path.exists(full_path) and should_load:
      print('Loading data set from %s...' % (full_path))
      train = np.load(full_path + 'train.npy')
      dev = np.load(full_path + 'dev.npy')
      test = np.load(full_path + 'test.npy')
   else:
      length = data.shape[0]
      print('total number of clusters in data is %d consisting of %d waveforms' % (length, count_waveforms(data)))
   
      per_dev += per_train
      
      if inds == []:
         np.random.shuffle(data)
      else:
         data = data[inds]
      train = data[:math.floor(length * per_train)]
      dev = data[math.floor(length * per_train): math.floor(length * per_dev)]
      test = data[math.floor(length * per_dev):]

      if path != None:
         try:
            if not os.path.exists(full_path):
               os.mkdir(full_path)
         except OSError:
            print ("Creation of the directory %s failed, not saving set" % full_path)
         else:
            print ("Successfully created the directory %s now saving data set" % full_path)
            np.save(full_path + 'train', train)
            np.save(full_path + 'dev', dev)
            np.save(full_path + 'test', test)

   num_clusters = data.shape[0]
   num_wfs = count_waveforms(data)
   
   print_data_stats(train, 'train', num_clusters, num_wfs)
   print_data_stats(dev, 'dev', num_clusters, num_wfs)
   print_data_stats(test, 'test', num_clusters, num_wfs)
   
   return train, dev, test

def print_data_stats(data, name, total_clusters, total_waveforms):
   """
   This function prints various statistics about the given set
   pyr == pyramidal ; in == interneuron; ut == untagged ; wfs == waveforms ; clstr == cluster
   """
   num_clstr = data.shape[0]
   num_wfs = count_waveforms(data)
   clstr_ratio = num_clstr / total_clusters
   wfs_ratio = num_wfs / total_waveforms
   print('Total number of clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
         % (name, num_clstr, 100 * clstr_ratio, num_wfs, 100 * wfs_ratio))

   pyr_clstrs = data[get_inds(data, 1)]
   num_pyr_clstr = pyr_clstrs.shape[0]
   ratio_pyr_clstr = num_pyr_clstr / total_clusters
   num_pyr_wfs = count_waveforms(pyr_clstrs)
   pyr_wfs_ratio = num_pyr_wfs / num_wfs
   print('Total number of  pyramidal clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
      % (name, num_pyr_clstr, 100 * ratio_pyr_clstr, num_pyr_wfs, 100 * pyr_wfs_ratio))
   
   in_clstrs = data[get_inds(data, 0)]
   num_in_clstr = in_clstrs.shape[0]
   ratio_in_clstr = num_in_clstr / total_clusters
   num_in_wfs = count_waveforms(in_clstrs)
   in_wfs_ratio = num_in_wfs / num_wfs
   print('Total number of  interneurons clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
      % (name, num_in_clstr, 100 * ratio_in_clstr, num_in_wfs, 100 * in_wfs_ratio))
   
   ut_clstrs = data[get_inds(data, -1)]
   num_ut_clstr = ut_clstrs.shape[0]
   ratio_ut_clstr = num_ut_clstr / total_clusters
   num_ut_wfs = count_waveforms(ut_clstrs)
   ut_wfs_ratio = num_ut_wfs / num_wfs
   print('Total number of  untagged clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
      % (name, num_ut_clstr, 100 * ratio_ut_clstr, num_ut_wfs, 100 * ut_wfs_ratio))        

def get_inds(data, label):
   inds = []
   for ind, cluster in enumerate(data):
      if label >= 0:
         if cluster[0, -1] == label:
            inds.append(ind)
      else:
         if cluster[0, -1] < 0:
            inds.append(ind)
   return inds

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

