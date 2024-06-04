#%%
# Import libraries
import pandas as pd

#%% Construcción de un dendrograma utilizando un método de agrupamiento jerárquico
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/Star Classification.csv')

# Paso 1: Seleccionar las características predictivas
X = data.drop(columns=['obj_ID', 'spec_obj_ID', 'class', 'plate', 'MJD'])

# Paso 2: Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3: Calcular la matriz de distancias y aplicar el método de agrupamiento jerárquico
Z = linkage(X_scaled, method='complete')

# Paso 4: Visualizar el dendrograma
plt.figure(figsize=(10, 7))
plt.title("Dendrograma del Agrupamiento Jerárquico")
plt.xlabel("Índice de la muestra")
plt.ylabel("Distancia")
dendrogram(Z)
plt.show()
