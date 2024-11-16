from mimetypes import inited
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Définir les moyennes et la covariance pour les deux classes
mean = np.array([0,0])
mean2 = np.array([10,10])
cov = np.array([[1,0.5],[0.5,1]])

# Générer les données pour chaque classe
Xclass1 = (np.random.multivariate_normal(mean,cov,128))
Xclass2 = (np.random.multivariate_normal(mean2,cov,128))
X = np.vstack((Xclass1,Xclass2))

#plt.plot(Xclass1[:,0],Xclass1[:,1], "o", label = 'Individu', color = 'red')
#plt.plot(Xclass1[:,0],Xclass2[:,1], "s", label = 'Individu2', color = 'blue')

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

kmeans = [ KMeans2, KMeans3, KMeans4, KMeans5, KMeans6]

plt.figure(figsize = (18,12))

# Tracer les points avec des symboles et couleurs différents pour chaque classe (KMeans2)
plt.subplot(2,3,1)
plt.scatter(X[KMeans2.labels_ == 0,0],X[KMeans2.labels_ == 0,1], c='cyan', label="Cluster 1 (k-means2)", marker='o')
plt.scatter(X[KMeans2.labels_ == 1, 0], X[KMeans2.labels_ == 1, 1], c='red', label="Cluster 2 (K-means)", marker='s')
plt.scatter(KMeans2.cluster_centers_[:, 0], KMeans2.cluster_centers_[:, 1], c='black', label="Centres des clusters", marker='o')

plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("KMeans2")


plt.subplot(2, 3, 2)
plt.scatter(X[true_labels == 0, 0], X[true_labels == 0, 1], c='red', label="Classe 1 (Vraie)", marker='s')
plt.scatter(X[true_labels == 1, 0], X[true_labels == 1, 1], c='cyan', label="Classe 2 (Vraie)", marker='o')

## Afficher la légende et les axes
plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("Vraie classification")


# Tracer les points avec des symboles et couleurs différents pour chaque classe (KMeans3)
plt.subplot(2, 3, 3)
plt.scatter(X[KMeans3.labels_ == 0, 0], X[KMeans3.labels_ == 0, 1], c='cyan', label="Cluster 1 (k-means3)", marker='o')
plt.scatter(X[KMeans3.labels_ == 1, 0], X[KMeans3.labels_ == 1, 1], c='red', label="Cluster 2 (k-means3)", marker='s')
plt.scatter(KMeans3.cluster_centers_[:, 0], KMeans3.cluster_centers_[:, 1], c='black', label="Centres des clusters", marker='o')

plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("KMeans3")

# Graphique 3 : KMeans4
plt.subplot(2, 3, 4)
plt.scatter(X[KMeans4.labels_ == 0, 0], X[KMeans4.labels_ == 0, 1], c='cyan', label="Cluster 1 (KMeans4)", marker='o')
plt.scatter(X[KMeans4.labels_ == 1, 0], X[KMeans4.labels_ == 1, 1], c='red', label="Cluster 2 (KMeans4)", marker='s')
plt.scatter(KMeans4.cluster_centers_[:, 0], KMeans4.cluster_centers_[:, 1], c='black', label="Centres des clusters", marker='o')
plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("KMeans4")

# Graphique 4 : KMeans5
plt.subplot(2, 3, 5)
plt.scatter(X[KMeans5.labels_ == 0, 0], X[KMeans5.labels_ == 0, 1], c='cyan', label="Cluster 1 (KMeans5)", marker='o')
plt.scatter(X[KMeans5.labels_ == 1, 0], X[KMeans5.labels_ == 1, 1], c='red', label="Cluster 2 (KMeans5)", marker='s')
plt.scatter(KMeans5.cluster_centers_[:, 0], KMeans5.cluster_centers_[:, 1], c='black', label="Centres des clusters", marker='o')
plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("KMeans5")

# Graphique 5 : KMeans6
plt.subplot(2, 3, 6)
plt.scatter(X[KMeans6.labels_ == 0, 0], X[KMeans6.labels_ == 0, 1], c='cyan', label="Cluster 1 (KMeans6)", marker='o')
plt.scatter(X[KMeans6.labels_ == 1, 0], X[KMeans6.labels_ == 1, 1], c='red', label="Cluster 2 (KMeans6)", marker='s')
plt.scatter(KMeans6.cluster_centers_[:, 0], KMeans6.cluster_centers_[:, 1], c='black', label="Centres des clusters", marker='o')
plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("KMeans6")

# Afficher les graphiques
plt.tight_layout()
plt.show()

k = 2
for i in kmeans:
    ars = adjusted_rand_score(true_labels, i.labels_)
    print("Adjusted Rand Score de K = ", k, " score : ", ars)
    k += 1


