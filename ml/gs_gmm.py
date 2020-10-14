from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np

import NN_util
from feature_dropping_svm import remove_features

N = 10 # number of time to repeat each configuration in the grid search

def metric(model, features, labels, verbos = False):
    #TODO get additional chectaristics of the clusters (for each feature mean std or ste)
    total = len(features)
    total_pyr = 0
    total_in = 0
    model_clusters = {}
    errors = 0
    pyr_clusters, in_clusters = 0, 0
    total_pyr += len(labels[labels == 1])
    total_in += len(labels[labels == 0])
    preds = model.predict(features)
    for i, pred in enumerate(preds):
        if pred not in model_clusters:
            model_clusters[pred] = [labels[i]]
        else:
            model_clusters[pred].append(labels[i])
    for ind, cluster in enumerate(model_clusters):
        labels = model_clusters[cluster]
        labels = np.asarray(labels)
        total = labels.shape[0]
        counter_pyr = len(labels[labels == 1])
        counter_in = len(labels[labels == 0])
        counter_ut = len(labels[labels < 0])
        if verbos:
            print('In cluster %d there are %d examples, of which %d are pyramidal (%.2f), %d are interneurons (%.2f) and %d are untagged (%.2f)' %
                  (ind + 1, total, counter_pyr, 100 * counter_pyr / total,  counter_in, 100 * counter_in / total, counter_ut,
                   100 * counter_ut / total))
        label = 0 if counter_in >= counter_pyr else 1 if counter_pyr != 0 else -1
        pyr_clusters += label
        in_clusters += 1 - label
        errors += counter_pyr if label == 0 else counter_in if label == 1 else 0

    print()
    print('Total of %d pyramidal cluster(s) and %d interneuron cluster(s)' % (pyr_clusters, in_clusters))
    print('Total number of errors is %d, the average error is %.3f' % (errors, errors / len(model_clusters)))
        
    return errors, errors / len(model_clusters)

def run(pca_n_components = None, use_pca = False, use_scale = False,
        tsne_n_components = None, use_tsne = False, vis_tsne_feature1 = 0, vis_tsne_feature2 = 1,
        filter_fets = [], verbos = False):
    # create or load dataset, no need to partiotion the data
    per_train = 1.0
    per_dev = 0.0
    per_test = 0.0
    NN_util.create_datasets(per_train, per_dev, per_test, should_filter = False)
    dataset_location = '../data_sets/clustersData_default' + '_' + str(per_train) + str(per_dev) + str(per_test) + '/'
    data, _, _ = NN_util.get_dataset(dataset_location)

    # remove features if needed
    if filter_fets != []:
        filter_fets.append(-1)
        data = remove_features(filter_fets, data)

    pca = None
    scaler = None

    # squeeze the data and separate features and labels
    data_squeezed = NN_util.squeeze_clusters(data)
    data_features, data_labels = NN_util.split_features(data_squeezed)

    # scale data if needed
    if use_scale or use_pca:
        print('Scaling data...')
        scaler = StandardScaler()
        scaler.fit(data_features)
        data_features = scaler.transform(data_features)

    # if we need to use PCA dimension reduction, we will fit PCA and transform the data
    if use_pca: 
        if pca_n_components == None:
            raise Exception('If use_pca is True but no loading path is given, pca_n_components must be specified')
        if pca_n_components > data_features.shape[1]:
            raise Exception('Number of required components is larger than the number of features')
        if not use_scale:
            raise Warning('When Using PCA data will be scaled even if use_scale is set to False')
        pca = PCA(n_components = pca_n_components, svd_solver = 'full')
        print('Fitting PCA...')
        pca.fit(data_features)
        print('explained variance by PCA components is: ' + str(pca.explained_variance_ratio_))
        print('Transforming training data with PCA...')
        data_features = pca.transform(data_features)

    # if we need to use TSNE dimension reduction, we will fit TSNE and transform the data
    if use_tsne: 
        if tsne_n_components == None:
            raise Exception('If use_tsne is True but no loading path is given, tsne_n_components must be specified')
        if tsne_n_components > data_features.shape[1]:
            raise Exception('Number of required components is larger than the number of features')
        print('Fitting and transforming data using tSNE...')
        data_features = TSNE(n_components = tsne_n_components).fit_transform(data_features)
        if verbos:
            plt.scatter(data_features[data_labels == 1, vis_tsne_feature1], data_features[data_labels == 1, vis_tsne_feature2], label = 'pyramidal')
            plt.scatter(data_features[data_labels == 0, vis_tsne_feature1], data_features[data_labels == 0, vis_tsne_feature2], label = 'interneuron')
            plt.scatter(data_features[data_labels < 0, vis_tsne_feature1], data_features[data_labels < 0, vis_tsne_feature2], label = 'unlabeled')
            plt.legend()
            plt.show()

    # define the parameters we will check
    n_components_options = np.arange(2, 21)
    covariance_type_options = ['full', 'tied', 'diag', 'spherical']

    # define variables to hold data from the grid search
    best_bic = float('inf')
    best_bic_values = (0, 0)
    bics = []

    # actual grid search
    for covariance_type in covariance_type_options:
        bics_temp = []
        for n_components in n_components_options:
            bic = 0
            for i in range(N):
                clst = GaussianMixture(n_components = n_components, covariance_type = covariance_type, reg_covar = 0.01)
                clst.fit(data_features)
                bic += clst.bic(data_features) / N # get bic score, averaged for N repetiotions
            bics_temp.append(bic)

            # update best options if improved
            if bic < best_bic:
                best_bic = bic
                best_bic_values = (n_components, covariance_type)
        bics.append(bics_temp)

        # evaluate best n_components for the specific covariance_type
        best_n_components = np.argmin(np.asarray(bics_temp)) + 2
        print('Best BIC with %s covariance type was %.3f, achieved with %d components' % (covariance_type, min(bics_temp), best_n_components))
        clst = GaussianMixture(n_components = best_n_components, covariance_type = covariance_type, reg_covar = 0.01)
        clst.fit(data_features)
        metric(clst, data_features, data_labels, verbos = verbos)
        print()

        # create and save a plot showing the change of BIC by n_components
        plt.clf()
        plt.plot(n_components_options, bics_temp)
        title = 'BIC score by n_components with ' + covariance_type + ' covariance type'
        plt.title(title)
        plt.xlabel('n_components')
        plt.ylabel('BIC')
        plt.savefig('../graphs/' + title + '_' + str(use_tsne))

    print('Best BIC score was %d ' % best_bic)
    print('The vars for best BIC score were: n_components = %d ; covariance_type = %s' % best_bic_values)

    # (re)evaluate best parameters all in all
    print('Evaluating predictions...')
    n_components, covariance_type = best_bic_values
    clst = GaussianMixture(n_components = n_components, covariance_type = covariance_type, reg_covar = 0.01)
    clst.fit(data_features)
    metric(clst, data_features, data_labels, verbos = verbos)

if __name__ == "__main__":
    run(use_tsne = True, tsne_n_components = 2, filter_fets = [], verbos = True, use_scale = False)
