import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar el archivo CSV
df = pd.read_csv("C:/Users/vmjai/PycharmProjects/AIRLINE-STUDY/base_datos_2008.csv")

# Limpiar los datos eliminando filas con valores nulos en 'Distance' y 'ArrDelay'
df_cleaned = df.dropna(subset=['Distance', 'ArrDelay'])

# Filtrar los vuelos con distancia > 2000 km y retraso > 2 horas (120 minutos)
df_filtered = df_cleaned[(df_cleaned['Distance'] > 2000) & (df_cleaned['ArrDelay'] > 120)]

# Seleccionar las columnas de interés (Distance y ArrDelay)
X = df_filtered[['Distance', 'ArrDelay']]

# Aplicar KMeans para encontrar el número de clústeres
kmeans = KMeans(n_clusters=3, random_state=42)
df_filtered.loc[:, 'Cluster'] = kmeans.fit_predict(X)

# Filtrar el Cluster 2 y almacenarlo en un nuevo DataFrame
df_cluster_2 = df_filtered[df_filtered['Cluster'] == 2]

# Verificar los nombres de las columnas
print(df_cluster_2.columns)  # Verifica que 'DayOfWeek' esté en el DataFrame

# 1. Agrupar por 'DayOfWeek' y calcular el resumen estadístico para 'ArrDelay'
daily_stats_cluster_2 = df_cluster_2.groupby('DayOfWeek')['ArrDelay'].describe()

# 2. Mostrar los resultados de las estadísticas descriptivas
print(daily_stats_cluster_2)

# 3. Visualización: Graficar los retrasos promedio por día de la semana
plt.figure(figsize=(10, 6))
plt.plot(daily_stats_cluster_2.index, daily_stats_cluster_2['mean'], marker='o', label='Promedio')
plt.title('Retraso Promedio por Día de la Semana (Cluster 2: Vuelos más largos y con mayores retrasos)')
plt.xlabel('Día de la Semana')
plt.ylabel('Retraso Promedio (minutos)')
plt.xticks(daily_stats_cluster_2.index, ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
plt.grid(True)
plt.legend()
plt.show()
