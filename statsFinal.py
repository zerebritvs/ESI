# Realizado por Juan Antonio Pagés López
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lee el archivo csv y almacena los datos en un DataFrame
data = pd.read_csv('./ESI_CargaTrabajo_2223.csv')

# Calcula los valores estadísticos para la columna NrmlTaskCores
cores_count = len(data['NrmlTaskCores'])
cores_mean = np.mean(data['NrmlTaskCores'])
cores_median = np.median(data['NrmlTaskCores'])
cores_std = np.std(data['NrmlTaskCores'])
cores_cov = cores_std / cores_mean
cores_min = np.min(data['NrmlTaskCores'])
cores_max = np.max(data['NrmlTaskCores'])
cores_p75 = np.percentile(data['NrmlTaskCores'], 75)
cores_p25 = np.percentile(data['NrmlTaskCores'], 25)
cores_iqr = cores_p75 - cores_p25

# Imprime los valores estadísticos para la columna NrmlTaskCores
print("---------------------- NrmlTaskCores ----------------------")
print(f"Recuento: {cores_count}")
print(f"Media: {cores_mean:.6f}")
print(f"Mediana: {cores_median:.6f}")
print(f"Desviación estándar: {cores_std:.6f}")
print(f"CoV: {cores_cov:.6f}")
print(f"Mínimo: {cores_min:.6f}")
print(f"Máximo: {cores_max:.6f}")
print(f"Q1: {np.percentile(data['NrmlTaskCores'], 25):.6f}")
print(f"Q3: {np.percentile(data['NrmlTaskCores'], 75):.6f}")
print(f"IQR: {cores_iqr:.6f}")

# Calcula los valores estadísticos para la columna NrmlTaskMem
mem_count = len(data['NrmlTaskMem'])
mem_mean = np.mean(data['NrmlTaskMem'])
mem_median = np.median(data['NrmlTaskMem'])
mem_std = np.std(data['NrmlTaskMem'])
mem_cov = mem_std / mem_mean
mem_min = np.min(data['NrmlTaskMem'])
mem_max = np.max(data['NrmlTaskMem'])
mem_p75 = np.percentile(data['NrmlTaskMem'], 75)
mem_p25 = np.percentile(data['NrmlTaskMem'], 25)
mem_iqr = mem_p75 - mem_p25

# Imprime los valores estadísticos para la columna NrmlTaskMem
print("---------------------- NrmlTaskMem ----------------------")
print(f"Recuento: {mem_count}")
print(f"Media: {mem_mean:.6f}")
print(f"Mediana: {mem_median:.6f}")
print(f"Desviación estándar: {mem_std:.6f}")
print(f"CoV: {mem_cov:.6f}")
print(f"Mínimo: {mem_min:.6f}")
print(f"Máximo: {mem_max:.6f}")
print(f"Q1: {np.percentile(data['NrmlTaskMem'], 25):.6f}")
print(f"Q3: {np.percentile(data['NrmlTaskMem'], 75):.6f}")
print(f"IQR: {mem_iqr:.6f}")

# Crea boxplots para las columnas NrmlTaskCores y NrmlTaskMem
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].boxplot(data['NrmlTaskCores'])
axs[0].set_title('Boxplot completo NrmlTaskCores')
axs[0].set_ylabel('NrmlTaskCores')
axs[1].boxplot(data['NrmlTaskMem'])
axs[1].set_title('Boxplot completo NrmlTaskMem')
axs[1].set_ylabel('NrmlTaskMem')
plt.show()

# Crea histogramas para las columnas NrmlTaskCores y NrmlTaskMem
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(data['NrmlTaskCores'], bins=40)
axs[0].set_title('Histograma completo NrmlTaskCores')
axs[0].set_xlabel('NrmlTaskCores')
axs[0].set_ylabel('Frecuencia')
axs[1].hist(data['NrmlTaskMem'], bins=40)
axs[1].set_title('Histograma completo NrmlTaskMem')
axs[1].set_xlabel('NrmlTaskMem')
axs[1].set_ylabel('Frecuencia')
plt.show()

print("---------------------- ParentID ----------------------")

recuento_parentID = data['ParentID'].nunique()
frecuencia = data['ParentID'].value_counts()
padres_repetidos = frecuencia.head(10)
print("Recuento de ParentIDs:", recuento_parentID)
# Imprimir los resultados de los 10 padres más repetidos
print("ID de los 10 primeros padres más repetidos:")
print(padres_repetidos)

# Calcula las estadísticas para cada JobType
for job_type in data['JobType'].unique():
    
    print(f"\n---------------------- Job Type {job_type} ----------------------")
    subset = data[data['JobType'] == job_type]
    print(f"Recuento: {len(subset)} ({(len(subset)/len(data))*100:.6f} %)")
    print(f"Media NrmlTaskCores: {subset['NrmlTaskCores'].mean():.6f}")
    print(f"Mediana NrmlTaskCores: {subset['NrmlTaskCores'].median():.6f}")
    print(f"Desviación estándar NrmlTaskCores: {subset['NrmlTaskCores'].std():.6f}")
    print(f"CoV NrmlTaskCores: "
        f"{subset['NrmlTaskCores'].std() / subset['NrmlTaskCores'].mean():.6f}")
    print(f"Mínimo NrmlTaskCores: {subset['NrmlTaskCores'].min():.6f}")
    print(f"Máximo NrmlTaskCores: {subset['NrmlTaskCores'].max():.6f}")
    q1 = subset['NrmlTaskCores'].quantile(0.25)
    q3 = subset['NrmlTaskCores'].quantile(0.75)
    print(f"Q1 NrmlTaskCores: {subset['NrmlTaskCores'].quantile(0.25):.6f}")
    print(f"Q3 NrmlTaskCores: {subset['NrmlTaskCores'].quantile(0.75):.6f}")
    iqr = q3 - q1
    print(f"IRQ NrmlTaskCores: {iqr:.6f}")
    
    print(f"\nMedia NrmlTaskMem: {subset['NrmlTaskMem'].mean():.6f}")
    print(f"Mediana NrmlTaskMem: {subset['NrmlTaskMem'].median():.6f}")
    print(f"Desviación estándar NrmlTaskMem: {subset['NrmlTaskMem'].std():.6f}")
    print(f"CoV NrmlTaskMem: "
        f"{subset['NrmlTaskMem'].std()/subset['NrmlTaskMem'].mean():.6f}")
    print(f"Mínimo NrmlTaskMem: {subset['NrmlTaskMem'].min():.6f}")
    print(f"Máximo NrmlTaskMem: {subset['NrmlTaskMem'].max():.6f}")
    q1 = subset['NrmlTaskMem'].quantile(0.25)
    q3 = subset['NrmlTaskMem'].quantile(0.75)
    print(f"Q1 NrmlTaskMem: {subset['NrmlTaskMem'].quantile(0.25):.6f}")
    print(f"Q3 NrmlTaskMem: {subset['NrmlTaskMem'].quantile(0.75):.6f}")
    iqr = q3 - q1
    print(f"IRQ NrmlTaskMem: {iqr:.6f}")
    
# Crea una figura con 4 boxplots para la variable NrmlTaskMem
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

subset_0 = data[data['JobType'] == 0]['NrmlTaskCores']
subset_1 = data[data['JobType'] == 1]['NrmlTaskCores']
subset_2 = data[data['JobType'] == 2]['NrmlTaskCores']
subset_3 = data[data['JobType'] == 3]['NrmlTaskCores']

axs.boxplot([subset_0, subset_1, subset_2, subset_3])
axs.set_xticklabels(['JobType 0', 'JobType 1', 'JobType 2', 'JobType 3'])
axs.set_ylabel('NrmlTaskCores')
axs.set_title('Boxplots NrmlTaskCores')
plt.show()

# Crea una figura con 4 boxplots para la variable NrmlTaskMem
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

subset_0 = data[data['JobType'] == 0]['NrmlTaskMem']
subset_1 = data[data['JobType'] == 1]['NrmlTaskMem']
subset_2 = data[data['JobType'] == 2]['NrmlTaskMem']
subset_3 = data[data['JobType'] == 3]['NrmlTaskMem']

axs.boxplot([subset_0, subset_1, subset_2, subset_3])
axs.set_xticklabels(['JobType 0', 'JobType 1', 'JobType 2', 'JobType 3'])
axs.set_ylabel('NrmlTaskMem')
axs.set_title('Boxplots NrmlTaskMem')
plt.show()

# Crea una figura y un conjunto de ejes para cada histograma
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Itera a través de los cuatro tipos de trabajo y crear un histograma para cada uno
for i in range(4):
    row = i // 2
    col = i % 2
    ax = axs[row][col]
    dataJobType = data[data['JobType'] == i]['NrmlTaskCores']
    ax.hist(dataJobType, bins=40)
    ax.set_title('Histograma JobType {}'.format(i))
    ax.set_xlabel('NrmlTaskCores')
    ax.set_ylabel('Frecuencia')  
    
# Ajusta el espacio entre subplots
plt.subplots_adjust(hspace=0.4, wspace=0.3)  
plt.show()

# Crea una figura y un conjunto de ejes para cada histograma
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Itera a través de los cuatro tipos de trabajo y crear un histograma para cada uno
for i in range(4):
    row = i // 2
    col = i % 2
    ax = axs[row][col]
    dataJobType = data[data['JobType'] == i]['NrmlTaskMem']
    ax.hist(dataJobType, bins=40)
    ax.set_title('Histograma JobType {}'.format(i))
    ax.set_xlabel('NrmlTaskMem')
    ax.set_ylabel('Frecuencia')

# Ajusta el espacio entre subplots
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()

# Imprime la correlación entre NrmlTaskCores y NrmlTaskMem
corr = data['NrmlTaskCores'].corr(data['NrmlTaskMem'])
print(f"\nCorrelación de NrmlTaskCores y NrmlTaskMem: {corr:.6f}")

# Define los colores para cada tipo de trabajo
colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}

# Crea una figura y cuatro ejes para cada tipo de trabajo
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Itera sobre los tipos de trabajo y sus ejes correspondientes
for job_type, ax in zip([0, 1, 2, 3], axs.flatten()):
    # Selecciona los puntos correspondientes al tipo de trabajo
    job_data = data[data['JobType'] == job_type]
    # Dibuja los puntos con el color correspondiente en el eje correspondiente
    ax.scatter(job_data['NrmlTaskCores'], job_data['NrmlTaskMem'], color=colors[job_type],
    label=job_type)
    ax.set_title(f'Scatterplot JobType {job_type}')
    ax.set_xlabel('NrmlTaskCores')
    ax.set_ylabel('NrmlTaskMem')
    ax.legend()
    
# Ajusta la disposición de los gráficos y mostrarlos
plt.tight_layout()
plt.show()
