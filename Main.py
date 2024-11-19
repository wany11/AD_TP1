# Importation des bibliothèques nécessaires
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

# Chargement des données
data_temperature = pd.read_csv("temperatures.csv", sep=";", decimal=".", header=0, index_col=0)

# Suppression des colonnes non pertinentes
data = data_temperature.drop(columns=['Region', 'Moyenne', 'Amplitude', 'Latitude', 'Longitude'])

# Normalisation des données
data_scaled = scale(data)

# Calcul de la matrice de dissimilarité
distance_matrix = pdist(data_scaled, metric='euclidean')

# Construction de l'arbre (méthode ward utilisée ici)
linkage_matrix = linkage(distance_matrix, method='ward')

# Visualisation de l'arbre
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=data.index, leaf_rotation=90, leaf_font_size=10)
plt.title("Dendrogramme - Classification Hiérarchique Ascendante")
plt.xlabel("Villes")
plt.ylabel("Distance")
plt.show()
