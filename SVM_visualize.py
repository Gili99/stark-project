import numpy as np
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h = 5):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out

def visualize_svm(data, targets, clf, h, feature1 = 0, feature2 = 1):
    X = data
    y = targets

    # title for the plot
    title = 'SVC with RBF kernel'

    X0, X1 = X[:, feature1], X[:, feature2]
    xx, yy = make_meshgrid(X0, X1, h = h)

    plot_contours(clf, xx, yy, cmap = plt.cm.coolwarm, alpha = 0.8)
    plt.scatter(X0, X1, c = y, cmap = plt.cm.coolwarm, edgecolors = 'k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)

    plt.show()
