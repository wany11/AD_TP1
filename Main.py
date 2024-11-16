import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage as CAH, fcluster
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


# Choisir le seuil automatiquement avec Z_complete[:, 2]
seuil_auto = np.max(Z_complete[:, 2]) - 0.000001

# Afficher le dendrogramme avec le seuil automatique
plt.figure(figsize=(12, 6))
dendrogram(Z_complete, color_threshold=seuil_auto)
plt.axhline(y=seuil_auto, color='red', linestyle='--', lw=1, label=f"Seuil automatique ({seuil_auto:.2f})")
plt.title("Dendrogramme CAH (Seuil Automatique)")
plt.xlabel("Échantillons")
plt.ylabel("Distance Euclidienne")
plt.legend()
plt.grid(True)
plt.show()

# Clustering pour K=3 clusters en utilisant le seuil déterminé
groupes_cah = fcluster(Z_complete, t=seuil, criterion='distance')

# Afficher les groupes avec 3 couleurs distinctes
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=groupes_cah, cmap='viridis')
plt.title("Clustering CAH - K=3")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()


# Calcul des différentes méthodes de linkage pour K=2 clusters
methods = ['single', 'complete', 'average', 'ward']
Z_methods = {method: CAH(X, method=method, metric="euclidean") for method in methods}

# Afficher les dendrogrammes pour les 4 méthodes sur un seul graphique
plt.figure(figsize=(12, 12))

for i, method in enumerate(methods, 1):
    plt.subplot(2, 2, i)
    dendrogram(Z_methods[method], color_threshold=0)
    plt.title(f"Dendrogramme - {method.capitalize()} Linkage")
    plt.xlabel("Échantillons")
    plt.ylabel("Distance Euclidienne")

plt.tight_layout()
plt.show()

# Calculer l'ARI pour chaque méthode avec K=2
seuil_k2 = Z_methods['complete'][-2, 2] - 0.000001
groupes_cah_k2 = fcluster(Z_methods['complete'], t=seuil_k2, criterion='distance')
ari_scores = {}
for method in methods:
    # Déterminer le seuil pour K=2 clusters en utilisant la méthode
    seuil_k2 = Z_methods[method][-2, 2] - 0.000001  # Seuil juste avant la fusion des deux derniers clusters
    # Obtenir les groupes en utilisant fcluster
    groupes_cah_k2 = fcluster(Z_methods[method], t=seuil_k2, criterion='distance')

    # Calculer l'ARI
    ari_scores[method] = adjusted_rand_score(true_labels, groupes_cah_k2)

# Afficher les résultats de l'ARI pour chaque méthode
for method, ari in ari_scores.items():
    print(f"Adjusted Rand Index pour la méthode '{method.capitalize()}': {ari:.3f}")
