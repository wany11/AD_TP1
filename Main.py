import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import progress
from sklearn.cluster import KMeans

def coalescence(x, K, g):
    # Nombre d'individus
    N = x.shape[0]

    # Initialisation des classes et des centres finaux
    clas = np.zeros(N, dtype=int)
    g2 = g.copy()

    # Convergence : boucle jusqu'à ce que les centres ne changent plus
    while True:
        # Étape 1 : Assignation des individus aux centres les plus proches
        for i in range(N):
            distances = np.linalg.norm(x[i, :] - g2, axis=1)  # Distances à chaque centre
            clas[i] = np.argmin(distances)  # Index du centre le plus proche

        # Étape 2 : Mise à jour des centres de gravité
        new_g2 = np.zeros_like(g2)
        for k in range(K):
            points_in_cluster = x[clas == k]  # Points appartenant au cluster k
            if len(points_in_cluster) > 0:  # Éviter les clusters vides
                new_g2[k, :] = np.mean(points_in_cluster, axis=0)

        # Arrêt si les centres ne changent plus
        if np.allclose(g2, new_g2):
            break
        g2 = new_g2  # Mise à jour des centres

    return clas, g2

# Génération des données
# Classe 1 : N2((2, 2)^T, 2I2)
mean1 = np.array([2, 2])
cov1 = 2 * np.eye(2)  # 2I2
class1 = np.random.multivariate_normal(mean1, cov1, 128)

# Classe 2 : N2((-4, -4)^T, 6I2)
mean2 = np.array([-4, -4])
cov2 = 6 * np.eye(2)  # 6I2
class2 = np.random.multivariate_normal(mean2, cov2, 128)

# Ensemble d'apprentissage
X = np.vstack((class1, class2))

# Sélection aléatoire de deux individus comme centres initiaux
initial_centers = X[np.random.choice(X.shape[0], 2, replace=False)]

# Application de l'algorithme de coalescence
clas, final_centers = coalescence(X, K=2, g=initial_centers)

# Affichage des résultats
plt.figure(figsize=(10, 6))

# Données initiales
plt.scatter(class1[:, 0], class1[:, 1], label="Classe 1", color='blue', alpha=0.6)
plt.scatter(class2[:, 0], class2[:, 1], label="Classe 2", color='green', alpha=0.6)

# Centres initiaux
plt.scatter(initial_centers[:, 0], initial_centers[:, 1],
            label="Centres initiaux", color='orange', edgecolor='black', marker='X', s=200)

# Centres finaux
plt.scatter(final_centers[:, 0], final_centers[:, 1],
            label="Centres finaux", color='red', edgecolor='black', marker='X', s=200)

# Attribution des clusters
for k in range(2):
    cluster_points = X[clas == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.3, label=f"Cluster {k+1}")

plt.legend()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Résultats de l'algorithme de coalescence (K-means)")
plt.grid()
plt.show()
