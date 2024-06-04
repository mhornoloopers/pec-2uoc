#%%
# Import libraries
import pandas as pd

#%% Procesamiento de datos
# Importar datos
data = pd.read_csv('data/Star Classification.csv')

# Revisión de datos faltantes
numero_datos_faltantes = data.isnull().sum().sum()

# Revisión de datos duplicados
numero_datos_duplicados = data.duplicated().sum()
#%%

#%% Normalización de datos
from sklearn.preprocessing import StandardScaler

# Seleccionamos solo las columnas numéricas para la normalización
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Excluimos las columnas de identificación que no son características predictivas
numerical_features.remove('obj_ID')
numerical_features.remove('spec_obj_ID')
numerical_features.remove('plate')
numerical_features.remove('MJD')

# Normalizamos las características numéricas
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
#%%

#%% Codificación de variables categóricas
data['class'] = data['class'].map({'GALAXY': 0, 'STAR': 1, 'QSO': 2})
#%%

#%% Algoritmo de clasificación
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Separar características y variable objetivo
X = data.drop(columns=['obj_ID', 'spec_obj_ID', 'class', 'plate', 'MJD'])
y = data['class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la exactitud
accuracy = accuracy_score(y_test, y_pred)
accuracy
#%%

#%% Métricas necesarias para evaluar el modelo de clasificación
from sklearn.metrics import classification_report, confusion_matrix

# Reporte de clasificación
report = classification_report(y_test, y_pred, target_names=['GALAXY', 'STAR', 'QSO'])

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

report, conf_matrix
#%%
