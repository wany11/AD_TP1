from mimetypes import inited

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import random
import os

from sklearn.metrics import adjusted_rand_score

mean = np.array([0,0])
mean2 = np.array([5,5])
cov = np.array([[1,0.5],[0.5,1]])

Xclass1 = (np.random.multivariate_normal(mean,cov,128))
Xclass2 = (np.random.multivariate_normal(mean2,cov,128))
X = np.vstack((Xclass1,Xclass2))

plt.plot(Xclass1[:,0],Xclass1[:,1], "o", label = 'Individu', color = 'red')
plt.plot(Xclass1[:,0],Xclass2[:,1], "s", label = 'Individu2', color = 'blue')

true_labels = np.hstack((np.zeros(len(Xclass1)), np.ones(len(Xclass2))))

#plt.scatter(Xclass1[:,0],Xclass1[:,1], label = 'Classe 1', color = 'red')

KMeans = KMeans(n_clusters=100, n_init=10, init='k-means++')
KMeans.fit(X)

plt.figure(figsize = (12,6))

plt.subplot(1,2,1)
plt.scatter(X[KMeans.labels_ == 0,0],X[KMeans.labels_ == 0,1], c='blue', label="Cluster 1 (k-means", marker='o')
plt.scatter(X[KMeans.labels_ == 1, 0], X[KMeans.labels_ == 1, 1], c='red', label="Cluster 2 (K-means)", marker='s')
plt.scatter(KMeans.cluster_centers_[:, 0], KMeans.cluster_centers_[:, 1], c='green', label="Centres des clusters", marker='x')

plt.subplot(1, 2, 2)
plt.scatter(X[true_labels == 0, 0], X[true_labels == 0, 1], c='blue', label="Classe 1 (Vraie)", marker='o')
plt.scatter(X[true_labels == 1, 0], X[true_labels == 1, 1], c='red', label="Classe 2 (Vraie)", marker='s')
plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("Vraie classification")



#print(kmeans.labels_)

plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("Ensemble de test")
plt.show()


ars = adjusted_rand_score(true_labels, KMeans.labels_)
print(ars)
