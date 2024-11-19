import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import progress
from sklearn.cluster import KMeans
import math

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
            distances = [math.sqrt((x[i][0] - g2[k][0]) ** 2 + (x[i][1] - g2[k][1]) ** 2) for k in range(K)]
            clas[i] = distances.index(min(distances))
        # Étape 2 : Mise à jour des centres de gravité
        new_g2 = np.zeros_like(g2)
        for k in range(K):
            points_in_cluster = x[clas == k]
            if len(points_in_cluster) > 0:
                new_g2[k, :] = np.mean(points_in_cluster, axis=0)

        # Arrêt si les centres ne changent plus
        if np.allclose(g2, new_g2):
            break
        g2 = new_g2  # Mise à jour des centres

    return clas, g2

# Génération des données
mean = np.array([0,0])
mean2 = np.array([5,5])
cov = np.array([[1,0.5],[0.5,1]])

# Ensemble d'apprentissage
Xclass1 = (np.random.multivariate_normal(mean,cov,128))
Xclass2 = (np.random.multivariate_normal(mean2,cov,128))
X = np.vstack((Xclass1,Xclass2))

# Sélection aléatoire de deux individus comme centres initiaux
initial_centers = X[np.random.choice(X.shape[0], 2, replace=False)]

# Application de l'algorithme de coalescence
clas, final_centers = coalescence(X, K=2, g=initial_centers)

# Affichage des résultats
plt.figure(figsize=(10, 6))

# Données initiales
plt.scatter(Xclass1[:, 0], Xclass1[:, 1], label="Classe 1", color='blue', alpha=0.6)
plt.scatter(Xclass2[:, 0], Xclass2[:, 1], label="Classe 2", color='green', alpha=0.6)

# Centres initiaux
plt.scatter(initial_centers[:, 0], initial_centers[:, 1],
            label="Centres initiaux", color='orange', edgecolor='black', marker='X', s=200)

# Centres finaux
plt.scatter(final_centers[:, 0], final_centers[:, 1],
            label="Centres finaux", color='red', edgecolor='black', marker='X', s=200)


plt.legend()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Résultats de l'algorithme de coalescence (K-means)")
plt.grid()
plt.show()
