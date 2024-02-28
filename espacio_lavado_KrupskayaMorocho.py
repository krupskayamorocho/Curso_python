""" Nombre: Krupskaya Morocho"""
#Variable clave:    espacio_lavado     
#Población objetivo:   region == "Costa" 

# Importamos numpy para realizar operaciones numéricas eficientes.
import numpy as np

# Pandas nos permitirá trabajar con conjuntos de datos estructurados.
import pandas as pd

# Desde sklearn.model_selection importaremos funciones para dividir conjuntos de datos y realizar validación cruzada.
from sklearn.model_selection import train_test_split, KFold

# Utilizaremos sklearn.preprocessing para preprocesar nuestros datos antes de entrenar modelos de aprendizaje automático.
from sklearn.preprocessing import StandardScaler

# sklearn.metrics nos proporcionará métricas para evaluar el rendimiento de nuestros modelos.
from sklearn.metrics import accuracy_score

# statsmodels.api nos permitirá realizar análisis estadísticos más detallados y estimación de modelos.
import statsmodels.api as sm

# Por último, matplotlib.pyplot nos ayudará a visualizar nuestros datos y resultados.
import matplotlib.pyplot as plt

#Cálculo de estadísticas básicas, revisión de la base 

# Impotar los datos originales
datos = pd.read_csv("sample_endi_model_10p.txt", sep=";")

# Convertimos los códigos numéricos de las regiones en etiquetas más comprensibles
datos["region"] = datos["region"].apply(
    lambda x: 
    "Costa" if x == 1 else "Sierra" if x == 2 else "Oriente")

# Filtramos el df según la población objetivo 
datos = datos[datos['region'] == 'Costa']

# Selecionamos SOLO las filas no contengan valores NA 
datos = datos[~datos["dcronica"].isna()]

datos.isna().sum()

# Seleccionamos las variables de interes 
variables = ['n_hijos', 'sexo', 'condicion_empleo', 'espacio_lavado', 'region']

# Eliminamos las filas con valores nulos en cada variable
for i in variables:
    datos = datos[~datos[i].isna()]

# Visualizamos el dataframe limpio 
datos.info

""" Ejercicio 1: Exploración de Datos"""
# Calcular el numero de niños que se encuentran en la region sierra (poblacion objetivo)
num_ninos_poblacion_objetivo = (datos["region"] == "Costa").sum()

# Calcular el conteo de niños que viven en region sierra que cuentan con un espacio de lavado de manos (variable asignada)
count_espacio_lavado = (datos["espacio_lavado"] == 1).sum()

print("El número de niños que se encuentran en la region Costa:", num_ninos_poblacion_objetivo)
print("El numero de niños de la region Costa con espacio de lavado de manos:", count_espacio_lavado)

#VARIABLES TRANFORMADAS

# Definimos las variables categóricas y numéricas que utilizaremos en nuestro análisis
variables_categoricas = ['region', 'sexo', 'condicion_empleo', 'espacio_lavado']
variables_numericas = ['n_hijos']

# Creamos un transformador para estandarizar las variables numéricas y una copia de nuestros datos para no modificar el conjunto original
transformador = StandardScaler()
datos_esc = datos.copy()

# Estandarizamos las variables numéricas utilizando el transformador
datos_esc[variables_numericas] = transformador.fit_transform(datos_esc[variables_numericas])

# Convertimos las variables categóricas en variables dummy utilizando one-hot encoding
datos_dummies = pd.get_dummies(datos_esc, drop_first = True)

datos_dummies.info()

# Seleccionamos las variables predictoras (X) y la variable ohttps://www.youtube.com/watch?v=Hr-K0Eke5KU&pp=ygUP7Jyk7IOBICBsb3ZlIGlzbjetivo (y) para nuestro modelo
X = datos_dummies[['n_hijos', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años', 'espacio_lavado']]
y = datos_dummies["dcronica"]

# Definimos los pesos asociados a cada observación para considerar el diseño muestral
weights = datos_dummies['fexp_nino']

#(TRAIN) Y PRUBEA (TEST)

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

print(X_train.isna().sum())
X_train.dtypes


""" Ejercicio 2: Modelo Logit

Los resultados de la regresión logística indican que la variable "espacio_lavado" 
tiene un coeficiente estimado de -1.1347, lo que sugiere que la presencia de un 
espacio de lavado está asociada negativamente con la variable dependiente "dcronica",
que representa la desnutrición crónica. Este coeficiente tiene un valor p (P>|z|) 
significativamente menor que 0.05, lo que indica una relación estadísticamente 
significativa entre la presencia de un espacio de lavado y la desnutrición crónica.

Sin embargo, es importante tener en cuenta que los demás coeficientes no son estadísticamente
significativos, ya que sus valores p son mayores que 0.05. Esto sugiere que las variables 
"n_hijos", "sexo_Mujer", "condicion_empleo_Empleada", "condicion_empleo_Inactiva" y 
"condicion_empleo_Menor a 15 años" no tienen una asociación significativa con la 
desnutrición crónica en este modelo.

Además, la precisión promedio del modelo al ser probado con datos de prueba es del 1.52%,
mientras que al ser probado con datos de entrenamiento es del 0.25%. Esto sugiere que el 
modelo podría tener dificultades para generalizar a nuevos datos y puede estar sobreajustado.
La advertencia sobre el máximo de iteraciones superado sugiere que el modelo puede no haber 
convergido adecuadamente, lo que también podría afectar su validez. Por lo tanto, se necesitarían 
más análisis y posiblemente ajustes en el modelo para mejorar su desempeño y validez.
"""

#Uso del train

modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)

# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)

# Comparamos las predicciones con los valores reales
predictions_class == y_test
print("La precisión promedio del modelo testeando con datos test es", np.mean(predictions_class))
 

# Realizamos predicciones en el conjunto de entrenamiento
predictions_train = result.predict(X_train)

# Convertimos las probabilidades en clases binarias
predictions_train_class = (predictions_train > 0.5).astype(int)

# Comparamos las predicciones con los valores reales
predictions_train_class == y_train
print("La precisión promedio del modelo testeando con datos train es", np.mean(predictions_train_class))


# --------------- VALIDACIÓN CRUZADA --------------- #

# Supongamos que X_train, X_test, y_train, y_test son tus conjuntos de datos de entrenamiento y prueba

# Define el número de divisiones para la validación cruzada
kf = KFold(n_splits=100)

# Lista para almacenar las precisiones de cada pliegue de validación cruzada
accuracy_scores = []

# DataFrame para almacenar los coeficientes estimados en cada pliegue
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # Aleatorizamos los folds en las partes necesarias
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    
    try:
        result_reg = log_reg.fit()
    except np.linalg.LinAlgError:
        print("Singular matrix detected. Applying regularization.")
        # Intenta ajustar el modelo de nuevo con regularización
        result_reg = log_reg.fit_regularized()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

# Imprimimos la precisión promedio de validación cruzada
print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")

###################Ejercicio 3: Evaluación del Modelo con Datos Filtrados###################

################ Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)

#Histograma para visualizar la distribución de las puntuaciones de precisión
plt.hist(accuracy_scores, bins=30, edgecolor='black')

# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()


#Histograma para ver la distribución de los coeficientes estimados para la variable “n_hijos”

plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["n_hijos"]), color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["n_hijos"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["n_hijos"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

"""Responder a las Preguntas: 
Responde a las siguientes preguntas en comentarios de tu script: 
¿Qué sucede con la precisión promedio del modelo cuando se utiliza el conjunto de datos filtrado? 
(Incremento o disminuye ¿Cuanto?)

La precisión promedio de validación cruzada cuando se utiliza el conjunto de datos filtrado es de 0.76625 
en comparación a la del ejercicio anterior de 0.731372549019608 vemos que aumenta

¿Qué sucede con la distribución de los coeficientes beta en comparación con el ejercicio anterior? 
(Incrementa o disminuye ¿Cuanto?)

Se puede ver que la media de los coeficientes Beta aumenta de un 0.11 a un 0.20  """