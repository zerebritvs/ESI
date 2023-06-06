# Realizado por Juan Antonio Pagés López
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Carga el conjunto de datos
data = pd.read_csv('./ESI_CargaTrabajo_2223.csv')

# Selecciona las columnas necesarias para el clustering
X = data[['NrmlTaskCores', 'NrmlTaskMem']].values

# Ejecuta k-medias con 6 clusters
kmeans = KMeans(n_clusters=6, n_init=10)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Obtiene el número de elementos en cada cluster y los ordena
element_counts = pd.Series(labels).value_counts()
percentages = (element_counts / len(X)) * 100
sorted_labels = element_counts.index

# Imprime información de los clusters
print("---------------------- Info Clusters ----------------------")
print(f"Número total de elementos: {len(X)}")
print("Número de elementos en cada cluster (ordenado por cantidad):")
for label in sorted_labels:
    count = element_counts[label]
    percentage = percentages[label]
    centroid = centroids[label]
    print(f"Cluster {label}: {count} elementos ({percentage:.6f} %). Centroides: {centroid}")

# Gráfica los puntos y los centroides de los clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3,
    color='black')
plt.title('K-medias con 6 clusters')
plt.xlabel('NrmlTaskCores')
plt.ylabel('NrmlTaskMem')

# Añade la leyenda en el gráfico con los clusters ordenados por cantidad de elementos
legend_elements = []
for label in sorted_labels:
    color = plt.cm.rainbow(label / (len(sorted_labels) - 1))
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
            markerfacecolor=color, markersize=10))
plt.legend(handles=legend_elements, loc='upper right')
plt.show()






