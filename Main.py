from mimetypes import inited
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Définir les moyennes et la covariance pour les deux classes
mean = np.array([0,0])
mean2 = np.array([5,5])
cov = np.array([[1,0.5],[0.5,1]])

# Générer les données pour chaque classe
Xclass1 = (np.random.multivariate_normal(mean,cov,128))
Xclass2 = (np.random.multivariate_normal(mean2,cov,128))
X = np.vstack((Xclass1,Xclass2))

plt.plot(Xclass1[:,0],Xclass1[:,1], "o", label = 'Individu', color = 'red')
plt.plot(Xclass1[:,0],Xclass2[:,1], "s", label = 'Individu2', color = 'blue')

true_labels = np.hstack((np.zeros(len(Xclass1)), np.ones(len(Xclass2))))

#plt.scatter(Xclass1[:,0],Xclass1[:,1], label = 'Classe 1', color = 'red')

KMeans2 = KMeans(n_clusters=2, n_init=10, init='k-means++')
KMeans2.fit(X)
KMeans3 = KMeans(n_clusters=3, n_init=10, init='k-means++')
KMeans3.fit(X)
KMeans4 = KMeans(n_clusters=4, n_init=10, init='k-means++')
KMeans4.fit(X)
KMeans5 = KMeans(n_clusters=5, n_init=10, init='k-means++')
KMeans5.fit(X)
KMeans6 = KMeans(n_clusters=6, n_init=10, init='k-means++')
KMeans6.fit(X)

plt.figure(figsize = (12,6))

# Tracer les points avec des symboles et couleurs différents pour chaque classe
plt.subplot(1,2,1)
plt.scatter(X[KMeans2.labels_ == 0,0],X[KMeans2.labels_ == 0,1], c='blue', label="Cluster 1 (k-means", marker='o')
plt.scatter(X[KMeans2.labels_ == 1, 0], X[KMeans2.labels_ == 1, 1], c='red', label="Cluster 2 (K-means)", marker='s')
plt.scatter(KMeans2.cluster_centers_[:, 0], KMeans2.cluster_centers_[:, 1], c='green', label="Centres des clusters", marker='o')

plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("test")


plt.subplot(1, 2, 2)
plt.scatter(X[true_labels == 0, 0], X[true_labels == 0, 1], c='red', label="Classe 1 (Vraie)", marker='s')
plt.scatter(X[true_labels == 1, 0], X[true_labels == 1, 1], c='blue', label="Classe 2 (Vraie)", marker='o')

# # Afficher la légende et les axes

plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("Vraie classification")

plt.show()


ars = adjusted_rand_score(true_labels, KMeans2.labels_)
print(ars)