import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

img=np.float32(mpimg.imread('visage.bmp'))
img_data = img.reshape(-1, 3)  # Forme (256*256, 3)

# Définir le nombre de couleurs souhaité
k = 250  # Nombre de clusters (attention ne pas en mettre trop sinon temps très long)
kmeans = KMeans(n_clusters=k, random_state=0).fit(img_data)

# Obtenir les centres des clusters (nouvelles couleurs)
centers = np.uint8(kmeans.cluster_centers_)
# Obtenir les étiquettes des clusters pour chaque pixel
labels = kmeans.labels_

compressed_img_data = centers[labels]
compressed_img = compressed_img_data.reshape(img.shape)  # Reformer l'image originale

# Affichage des images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Image originale")
plt.imshow(np.uint8(img))  # Image d'origine

plt.subplot(1, 2, 2)
plt.title(f"Image compressée ({k} couleurs)")
plt.imshow(compressed_img / 255)  # Image compressée


plt.show()