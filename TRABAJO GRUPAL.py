## Haciendo uso de la visualización para contar una historia
## Trabajo grupal
# Melissa Chumaña
#Krupskaya Morocho

# importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

#Cargo la base de datos de esperanza de vida, mujeres.
df_countries = pd.read_excel("Esperanza de vida al nacer, mujeres (años).xls",sheet_name="Data")
print(df_countries)

df_index = pd.read_excel("Esperanza de vida al nacer, mujeres (años).xls",sheet_name="Data",skiprows=  3)
print(df_index)

# Filtrar los datos para América Latina
latin_america = df_index[df_index['Country Name'].isin(['Argentina', 'Brazil', 'Mexico', 'Chile', 'Colombia',
'Peru', 'Ecuador', 'Venezuela', 'Bolivia', 'Paraguay', 'Uruguay', 'Guatemala', 'Cuba', 'Honduras', 'Nicaragua',
'El Salvador', 'Costa Rica', 'Panama', 'Dominican Republic', 'Puerto Rico'])]

print( latin_america)

#PREGUNTA 1
# Calcular el promedio del indicador en el año 2020
esperanza_vida_2020 = df_index['2020'].mean()

print("El promedio de la esperanza de vida al nacer para mujeres en 2022 es:", esperanza_vida_2020)

#PREGUNTA 2
# Seleccionar solo las filas y columnas relevantes
latin_america_subset = latin_america.set_index('Country Name').loc['1960':'2022']

# Transponer el DataFrame
latin_america_transposed = latin_america_subset.transpose()

# Graficar la evolución de los indicadores para los países seleccionados
latin_america_transposed.plot(figsize=(12, 8))
plt.title('Evolución de los indicadores en países de América Latina')
plt.xlabel('Año')
plt.ylabel('Valor del Indicador')
plt.legend(title='País', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#PREGUNTA 3 
# Filtrar el DataFrame para incluir solo los últimos 5 años
df_filtrado = latin_america_transposed.loc[['2017', '2018', '2019', '2020', '2021']]

# Calcula la matriz de correlación
matriz_correlacion = df_filtrado.corr()

# Usa seaborn para crear un mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlacion, annot=True, cmap="YlGnBu")
plt.title('Mapa de Correlación entre los últimos 5 años de datos en América Latina')
plt.show()
