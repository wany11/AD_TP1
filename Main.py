# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

# Choisir le seuil automatiquement avec Z_complete[:, 2]
seuil = 10 - 0.00000001

# Visualisation de l'arbre
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=data.index, leaf_rotation=90, leaf_font_size=10, color_threshold=seuil)
plt.axhline(y=seuil, color='red', linestyle='--', lw=1, label=f"Seuil automatique ({seuil:.2f})")
plt.title("Dendrogramme - Classification Hiérarchique Ascendante")
plt.xlabel("Villes")
plt.ylabel("Distance")
plt.legend()
plt.show()

# Découper l'arbre en clusters (par exemple, 4 clusters)
num_clusters = 10
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Ajouter les clusters pour chaque ville
data_temperature['Cluster'] = clusters

Coord = data_temperature.loc[:, ['Latitude', 'Longitude']].values

nom_ville = list(data.index)

# Récupération des colonnes Latitude et Longitude pour la visualisation
latitude = data_temperature['Latitude']
longitude = data_temperature['Longitude']

# Visualisation des clusters avec Longitude et Latitude
plt.figure(figsize=(10, 6))
for cluster in data_temperature['Cluster'].unique():
    cluster_data = data_temperature[data_temperature['Cluster'] == cluster]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'Cluster {cluster}')
for i, txt in enumerate(nom_ville):
    plt.annotate(txt, (longitude.iloc[i], latitude.iloc[i]), fontsize=8, alpha=0.7)
plt.title("Clustering des villes en fonction de leur Longitude et Latitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# Appliquer K-Means avec un nombre de clusters défini
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, n_init=10, init='k-means++', random_state=42)
data_temperature['Cluster'] = kmeans.fit_predict(data_scaled)


# Visualisation des clusters
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = data_temperature[data_temperature['Cluster'] == cluster]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'Cluster {cluster + 1}')

# Ajouter les centres des clusters
centers = kmeans.cluster_centers_  # Ces centres sont dans l'espace des données normalisées
# centers_transformed = kmeans.inverse_transform(centers)  # Ramener les centres dans l'espace d'origine (si nécessaire)

# Annotation des villes
for i, txt in enumerate(data.index):
    plt.annotate(txt, (longitude.iloc[i], latitude.iloc[i]), fontsize=8, alpha=0.7)

plt.title("Clustering K-Means des villes en fonction de la Longitude et Latitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)

inertias = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', linestyle='-', color='b')
plt.title("Inertie en fonction du nombre de clusters K")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertie")
plt.xticks(k_values)
plt.grid(True)
plt.show()