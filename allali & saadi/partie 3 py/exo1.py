#################  Exercice 1 ###############
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