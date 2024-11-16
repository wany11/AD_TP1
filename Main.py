import numpy as np
import matplotlib.pyplot as plt

# Création des deux classes
mean1 = np.array([2, 2])
cov1 = np.array([[1, 0.5], [0.5, 1]])  # Matrice de covariance
class1 = np.random.multivariate_normal(mean1, cov1, 128)

mean2 = np.array([-4, -4])
cov2 = np.array([[1, 0.5], [0.5, 1]])
class2 = np.random.multivariate_normal(mean2, cov2, 128)

# Fusion des deux classes
data = np.vstack((class1, class2))
labels_true = np.array([0] * 128 + [1] * 128)

# Affichage
plt.scatter(class1[:, 0], class1[:, 1], label="Classe 1", alpha=0.7, color="red")
plt.scatter(class2[:, 0], class2[:, 1], label="Classe 2", alpha=0.7, color="blue")
plt.legend()
plt.title("Corpus de test : Deux classes gaussiennes")
plt.show()