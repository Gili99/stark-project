from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import seaborn as sns

import NN_util
from VIS_heatmap import create_heatmap

cs = np.logspace(-2.0, 4.0, num = 7)
gammas = np.logspace(-7.0, -1.0, num = 7)
grid = np.dstack(np.meshgrid(cs, gammas)).reshape(-1, 2)

def evaluate(clf, data):
    total_clusters, total_chunks = 0, 0
    correct_clusters, correct_chunks = 0, 0
    for cluster in data:
        features, labels = NN_util.split_features(cluster)
        label = labels[0] #as they are the same for all the cluster
        preds = clf.predict(features)
        prediction = round(preds.mean())
        correct_clusters += 1 if prediction == label else 0
        correct_chunks += preds[preds == label].shape[0]
        total_clusters += 1
        total_chunks += labels.shape[0]
    return correct_clusters / total_clusters, correct_chunks / total_chunks
        

per_train = 0.6
per_dev = 0.2
per_test = 0.2
NN_util.create_datasets(per_train, per_dev, per_test)
dataset_location = '../data_sets/clustersData_default' + '_' + str(per_train) + str(per_dev) + str(per_test) + '/'
train, dev, test = NN_util.get_dataset(dataset_location)
train_squeezed = NN_util.squeeze_clusters(train)
train_features, train_labels = NN_util.split_features(train_squeezed)

correct_clusters_p = []
best_clusters = 0
best_clusters_vals = (0, 0)
correct_chunks_p = []
best_chunks = 0
best_chunks_vals = (0, 0)


for c, gamma in tqdm(grid):
    clf = svm.SVC(kernel = 'rbf', class_weight = 'balanced', C = c, gamma = gamma)
    print('Fitting SVM with c = %.2f and gamma = %.7f...' % (c, gamma))
    clf.fit(train_features, train_labels)
    print('Evaluating classifier...')
    cclusters_p, cchunks_p = evaluate(clf, dev)
    correct_clusters_p.append(cclusters_p)
    correct_chunks_p.append(cchunks_p)
    if cclusters_p > best_clusters:
       best_clusters_vals = (c, gamma)
    if cchunks_p > best_chunks:
       best_chunks_vals = (c, gamma) 

print(correct_clusters_p)
print(correct_chunks_p)

print('Best cost and gamma according to cluster accuracy are cost: %.2f ; gamma: %.7f' % (best_clusters_vals[0], best_clusters_vals[1]))
print('Best cost and gamma according to chunk accuracy are cost: %.2f ; gamma: %.7f' % (best_chunks_vals[0], best_chunks_vals[1]))

correct_clusters_p = np.asarray(correct_clusters_p)
correct_chunks_p = np.asarray(correct_chunks_p)

create_heatmap(gammas, cs, 'Gamaa', 'Cost', 'Clusters accuracy by cost and gamma', correct_clusters_p.reshape(cs.shape[0], -1), path = 'graphs/')
create_heatmap(gammas, cs, 'Gamaa', 'Cost', 'Chunks accuracy by cost and gamma', correct_chunks_p.reshape(cs.shape[0], -1), path = 'graphs/')
