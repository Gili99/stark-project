from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

import NN_util


# get the matrix of features
data = NN_util.read_data('../clustersData', should_filter = True)
train, _, _ = NN_util.split_data(data, per_train = 1.0, per_dev = 0.0, per_test = 0.0)
train_squeezed = NN_util.squeeze_clusters(train)
train_features, train_labels = NN_util.split_features(train_squeezed)

# get only the relevant features
relevant_features = train_features[:, [6, 8]]
#plt.scatter(train_features[:, 0], train_features[:, 1])
#plt.show()

# Standardizing
scaler = StandardScaler()
stand_relevant = scaler.fit_transform(relevant_features)
#plt.scatter(stand_relevant[:, 0], stand_relevant[:, 1], c=train_labels)
#plt.show()

#run kmeans
clst = KMeans(n_clusters=2, random_state=0).fit_predict(stand_relevant)
#plt.scatter(stand_relevant[:, 0], stand_relevant[:, 1], c=clst)
#plt.show()

# compute the proportion in each cluster
res = np.zeros((2, 2))
for i, cluster in enumerate(clst):
    label = train_labels[i]
    
    # add the type of cell to the count
    if label == 0: # check if cell is interneuron
        res[cluster][0] += 1
    else:
        res[cluster][1] += 1

for i in range(res.shape[0]):
    print("In cluster %d there are %d interneurons and %d pyramidal" % (i, res[i][0], res[i][1]))
