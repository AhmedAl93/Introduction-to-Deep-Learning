from keras.datasets import mnist
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
import matplotlib as mpl

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_embedded = TSNE(n_components=2,verbose=2,init='pca').fit_transform(X_train[0:1000,:])
def convexHulls(points, labels):
# computing convex hulls for a set of points with asscoiated labels
    convex_hulls = []
    for i in range(10):
        convex_hulls.append(ConvexHull(points[labels==i,:]))
    return convex_hulls
# Function Call
convex_hulls= convexHulls(X_embedded, y_train[0:1000])
def best_ellipses(points, labels):
    # computing best fiiting ellipse for a set of points with asscoiated labels
    gaussians = []
    for i in range(10):
        gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels==i, :]))
    return gaussians
# Function Call
ellipses = best_ellipses(X_embedded, y_train[0:1000])

from sklearn.neighbors import NearestNeighbors
def neighboring_hit(points, labels):
    k = 6
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    txs = 0.0
    txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(len(points)):
        tx = 0.0
        for j in range(1,k+1):
            if (labels[indices[i,j]]== labels[i]):
                tx += 1
        tx /= k
        txsc[labels[i]] += tx
        nppts[labels[i]] += 1
        txs += tx
    for i in range(10):
        txsc[i] /= nppts[i]
    print txsc

    return txs / len(points)

go=neighboring_hit(X_embedded, y_train[0:1000])     

def visualization(points2D, labels, convex_hulls, ellipses ,projname, nh):
    points2D_c= []
    for i in range(10):
        points2D_c.append(points2D[labels==i, :])
  # Data Visualization
    cmap =cm.Vega10
    plt.figure(figsize=(3.841, 7.195), dpi=100)
    plt.set_cmap(cmap)
    plt.subplots_adjust(hspace=0.4 )
    plt.subplot(311)
    plt.scatter(points2D[:,0], points2D[:,1], c=labels,  s=3,edgecolors='none', cmap=cmap, alpha=1.0)
    plt.colorbar(ticks=range(10))
    plt.title("2D "+projname+" - NH="+str(nh*100.0))
    vals = [ i/10.0 for i in range(10)]
    sp2 = plt.subplot(312)
    for i in range(10):
        ch = np.append(convex_hulls[i].vertices,convex_hulls[i].vertices[0])
        sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-',label='$%i$'%i, color=cmap(vals[i]))
    plt.colorbar(ticks=range(10))
    plt.title(projname+" Convex Hulls")
 
    def plot_results(X, Y_, means, covariances, index, title, color):
        splot = plt.subplot(3, 1, 3)
        for i, (mean, covar) in enumerate(zip(means, covariances)):
          v, w = linalg.eigh(covar)
          v = 2. * np.sqrt(2.) * np.sqrt(v)
          u = w[0] / linalg.norm(w[0])
          # as the DP will not use every component it has access to
          # unless it needs it, we shouldn't plot the redundant
          # components.
          if not np.any(Y_ == i):
              continue
          plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha = 0.2)

          # Plot an ellipse to show the Gaussian component
          angle = np.arctan(u[1] / u[0])
          angle = 180. * angle / np.pi  # convert to degrees
          ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
          ell.set_clip_box(splot.bbox)
          ell.set_alpha(0.6)
          splot.add_artist(ell)
        plt.title(title)
    plt.subplot(313)
    for i in range(10):
        plot_results(points2D[labels==i, :], ellipses[i].predict(points2D[labels==i, :]), ellipses[i].means_,
        ellipses[i].covariances_, 0,projname+" fitting ellipses", cmap(vals[i]))

    plt.savefig(projname+".png", dpi=100)
    plt.show()                  
visualization(X_embedded, y_train[0:1000], convex_hulls, ellipses ,'tht', go)
