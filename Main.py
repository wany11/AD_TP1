import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage as CAH
from scipy.cluster.hierarchy import dendrogram

# Générer les données
mean = np.array([0, 0])
mean2 = np.array([5, 5])
cov = np.array([[1, 0.5], [0.5, 1]])

Xclass1 = np.random.multivariate_normal(mean, cov, 128)
Xclass2 = np.random.multivariate_normal(mean2, cov, 128)
X = np.vstack((Xclass1, Xclass2))

true_labels = np.hstack((np.zeros(len(Xclass1)), np.ones(len(Xclass2))))


# Calculer la CAH avec le linkage "complete"
Z_complete = CAH(X, method="complete", metric="euclidean")

# Déterminer le seuil pour K=3 clusters
# On choisit une hauteur juste avant la fusion en deux clusters.
seuil = Z_complete[-2, 2] - 0.000001

# Afficher le dendrogramme
plt.figure(figsize=(12, 6))
dendrogram(Z_complete, color_threshold=seuil)
plt.axhline(y=seuil, color='red', linestyle='--',lw=1, label=f"Seuil pour K=3 ({seuil:.2f})")
plt.title("Dendrogramme CAH (K=3 Clusters)")
plt.xlabel("Échantillons")
plt.ylabel("Distance Euclidienne")
plt.legend()
plt.grid(True)
plt.show()
