import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Définir les moyennes et la covariance pour les deux classes
mean1 = np.array([0, 0])
mean2 = np.array([5, 10])
cov = np.array([[1, 0.5], [0.5, 1]])

# Générer les données pour chaque classe
X_class1 = np.random.multivariate_normal(mean1, cov, 256)
X_class2 = np.random.multivariate_normal(mean2, cov, 256)

X = np.vstack((X_class1, X_class2))
true_labels = np.hstack((np.zeros(len(X_class1)), np.ones(len(X_class2))))

KMeans = KMeans(n_clusters=100, n_init=100, init='k-means++')
KMeans.fit(X)

# Tracer les points avec des symboles et couleurs différents pour chaque classe
plt.plot(X_class1[:, 0], X_class1[:, 1], 'o', label="Classe 1", color="blue")
plt.plot(X_class2[:, 0], X_class2[:, 1], 'o', label="Classe 2", color="red")
plt.plot(KMeans.cluster_centers_[:, 0], KMeans.cluster_centers_[:, 1], 'o', label="Cluster centers", color="green")

# # Afficher la légende et les axes
plt.legend()
plt.xlabel("Mesure 1")
plt.ylabel("Mesure 2")
plt.title("Ensemble de test avec deux classes")
plt.show()

ars = adjusted_rand_score(true_labels, KMeans.labels_)
print(ars)






