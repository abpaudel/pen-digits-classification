import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sympy.utilities.iterables import multiset_permutations


perm = np.array(list(multiset_permutations(range(8))))

thld = np.array([70,110,100,110,50,80,100,60,100,70])
#                0   1   2   3  4  5   6  7   8  9
with open('centroids.npy', 'rb') as f:
    centroids = np.load(f)

order=np.array([[0,4,2,7,1,5,3,6],#0
                [0,5,2,6,4,1,3,7],#1
                [4,2,7,0,5,1,3,6],#2
                [1,5,6,2,3,7,4,0],#3
                [7,2,6,1,0,3,5,4],#4
                [6,2,0,3,7,4,1,5],#5
                [5,2,3,7,0,4,1,6],#6
                [1,7,3,5,0,6,2,4],#7
                [6,4,0,1,7,3,5,2],#8
                [0,6,1,7,3,5,2,4]])#9


def plot_mnist(img, label=''):
    plt.imshow(img.reshape((28,28)), cmap='gray')
    plt.title(str(label))
    plt.xticks([])
    plt.yticks([])

def plot_coord(coord, marker='o', lim=(-1,101)):
    plt.scatter(coord[:,0], coord[:,1], marker=marker)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xticks([])
    plt.yticks([])

def img2coord(img, scale=100, pixval_min=1):
    r, c = np.where(img.reshape((28,28))>=pixval_min)
    x_coor, y_coor = (c*100/28).astype('int64'), ((28-r)*100/28).astype('int64')
    xycoors = np.concatenate((x_coor.reshape(-1,1), y_coor.reshape(-1,1)), axis=1)
    return xycoors

def mnist2clusters(img, pixval_min=100, scale=100):
    xycoors = img2coord(img, scale=scale, pixval_min=pixval_min)
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(xycoors)
    return kmeans, xycoors

def plot_clusters(kmeans, xycoors, marker='o', lim=(-1,101), only_centroids=False):
    pred = kmeans.predict(xycoors)
    if only_centroids:
        plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*',color='k')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.xticks([])
        plt.yticks([])
    else:   
        plt.scatter(xycoors[:,0], xycoors[:,1], c=pred, cmap='Dark2', marker=marker)
        plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*',color='k')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.xticks([])
        plt.yticks([])

def mse(c1, c2):
    return np.mean((c1-c2)**2)

def mse_centroids(centroid_indx, tmplt, centroids):
    return mse(tmplt, centroids[perm[centroid_indx]])

def minimize(tmplt, centroids):
    mse_val = [mse_centroids(i, tmplt, centroids) for i in range(len(perm))]
    return centroids[perm[np.argmin(mse_val)]], np.min(mse_val)

def feature_extractor(img, label, return_clusters=False):
    tmplt = centroids[label][order[label]]
    km, xy = mnist2clusters(img, pixval_min=thld[label])
    points8, _ = minimize(tmplt, km.cluster_centers_.astype('float64'))
    if return_clusters:
        return points8, km, xy
    return points8

def plot_exfeatures(coord, marker='--o', c='tab:blue',lim=(-1,101)):
    plt.plot(coord[:,0],coord[:,1], marker, color=c)
    plt.scatter(coord[0,0],coord[0,1], marker='s', color=c)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xticks([])
    plt.yticks([])

def get_boundary(coord):
    xmin = coord[:,0].min()
    xmax = coord[:,0].max()
    ymin = coord[:,1].min()
    ymax = coord[:,1].max()
    return xmin,ymin,xmax,ymax

def scale(coord):
    xmin,ymin,xmax,ymax = get_boundary(coord)
    x_o = (xmax + xmin)/2
    y_o = (ymax + ymin)/2
    coord = coord - [x_o,y_o]
    sc = 100/max(xmax-xmin, ymax-ymin)
    coord *= sc
    coord += 50
    return coord.astype('int64')