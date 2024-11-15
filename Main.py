import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples


# Définir les moyennes et la covariance pour les deux classes
mean = np.array([0,0])
mean2 = np.array([5,5])
cov = np.array([[1,0.5],[0.5,1]])

# Générer les données pour chaque classe
Xclass1 = (np.random.multivariate_normal(mean,cov,128))
Xclass2 = (np.random.multivariate_normal(mean2,cov,128))
X = np.vstack((Xclass1,Xclass2))

true_labels = np.hstack((np.zeros(len(Xclass1)), np.ones(len(Xclass2))))

inertias = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', linestyle='-', color='b')
plt.title("Inertie en fonction du nombre de clusters K")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertie")
plt.xticks(k_values)
plt.grid(True)
plt.show()
plt.figure(figsize = (12,6))
