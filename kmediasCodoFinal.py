# Realizado por Juan Antonio Pagés López
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carga los datos desde el archivo CSV
data = pd.read_csv('./ESI_CargaTrabajo_2223.csv')

# Extrae las columnas NrmlTaskMem y NrmlTaskCores de los datos
X = data[['NrmlTaskMem', 'NrmlTaskCores']]

# Inicializa las listas para guardar los valores de TWSS y k
twss_values = []
k_values = []

# Calcula TWSS para diferentes valores de k
for k in range(2, 11):
    # Inicializa el modelo de k-medias
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    
    # Calcula la suma de los errores cuadrados internos (TWSS)
    twss = kmeans.inertia_
    
    # Añade los valores de k y TWSS a las listas correspondientes
    k_values.append(k)
    twss_values.append(twss)

# Dibuja la gráfica del criterio del codo
plt.plot(k_values, twss_values, 'ro-')
plt.xlabel('Número de clusters k')
plt.ylabel('Total Within Sum of Squares (TWSS)')
plt.title('Criterio del Codo')
plt.show()
