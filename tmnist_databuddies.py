# ================= LABORATORIO DE DATOS: TRABAJO PRÁCTICO 2 ==================
# ============================ GRUPO E INTEGRANTES ============================
'''
 Data Buddies:
    -Eliseo Carriedo  (L.U.: 392/23)
    -Lila Fage (L.U.: 235/24)
    -Julian Laurido (L.U.: 1097/23 )
'''
# ================================== ÍNDICE ===================================

# Imports ------------------------------------------------------------------ 17
# Carga de datos ----------------------------------------------------------- 36
# Análisis exploratorio ---------------------------------------------------- 44
# Clasificación binaria ---------------------------------------------------- 63
# Clasificación multiclase ------------------------------------------------ 106

#%% IMPORTS ===================================================================
# ------------------------------- Archivos Py ---------------------------------
from exploratorio import *
from binario import *
from multiclase import *

# -------------------------------- Librerias ----------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from inline_sql import sql
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#%% CARGA DE DATOS ============================================================

carpeta = './'
data = pd.read_csv(carpeta+'TMNIST_Data.csv')
columnas_num = data.select_dtypes(include=[np.int64]).columns[1:]
data[columnas_num] = data[columnas_num].astype(np.int32)
del(columnas_num)

#%% ANALISIS EXPLORATORIO =====================================================
#&& a) ------------------------------------------------------------------------
    
no_importantes = std_data(data)[std_data(data) == 0].index
plot_std_data(data)

#%% b) ------------------------------------------------------------------------

# Promedio intensidad de cada pixel por número
plot_media_numero(data)
            
# Variabilidad de intensidad entre 1 y 4 y entre 1 y 0
plot_std_1y4_1y0(data)

#%% c) ------------------------------------------------------------------------

# Variabilidad y promedio de la clase 0
plot_numeros(data)

#%% CLASIFICACION BINARIA =====================================================
#&& a) ------------------------------------------------------------------------

# Muestras de ceros y unos
data_0y1 = data[(data['labels'] == 0) | (data['labels'] == 1)].iloc[:,1:].reset_index(drop=True)

# Verificamos que la cantidad de muestras esté balanceada
count_0y1 = sql^ """
                 SELECT labels, COUNT(*) AS cantidad
                 FROM data_0y1
                 GROUP BY labels
                 """

#%% b) ------------------------------------------------------------------------

# Descartamos los no importantes
std_0y1 = np.std(data_0y1.iloc[:,1:], axis = 0)
no_importantes_0y1 = std_0y1[std_0y1 == 0].index
importantes_0y1 = data_0y1.drop(no_importantes_0y1, axis = 1)

X = importantes_0y1.iloc[:,1:]
Y = importantes_0y1['labels']

# Separamos en Train y Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

#%% c) ------------------------------------------------------------------------

# Exactitud con 3 atributos constante
n_atributos = [3]*15
exactitud_0y1_por_atributos (n_atributos, X_train, X_test, Y_train, Y_test)

# Exactitud variando atributos entre 3 y 18
n_atributos = np.arange(3,19,1)
exactitud_0y1_por_atributos (n_atributos, X_train, X_test, Y_train, Y_test)

#%% d) ------------------------------------------------------------------------

# Exactitudes variando vecinos y atributos
n_atributos = np.arange(3,16,1)
k_vecinos = np.arange(3,16,1)
exactitudes = dataExactitudes(n_atributos, k_vecinos, X_train, X_test, Y_train, Y_test)

exactitudes_plot(exactitudes)

#&& Borrado -------------------------------------------------------------------

del (count_0y1, data_0y1, importantes_0y1, n_atributos, no_importantes_0y1, std_0y1, X, Y,X_train, X_test, Y_train, Y_test)

#%% CLASIFICACIÓN MULTICLASE ==================================================
#&& a) ------------------------------------------------------------------------

# Separado de información relevante
data_importante = data.drop(no_importantes, axis = 1)
X = data_importante.iloc[:,2:]
Y = data_importante.iloc[:,1]

# Separado de Desarrollo y Held-Out
X_dev, X_held, Y_dev, Y_held = train_test_split(X, Y, test_size=0.1, shuffle=False)

#%% b) ------------------------------------------------------------------------

# Separado en Train y Test
X_train, X_test, Y_train, Y_test = train_test_split(X_dev, Y_dev, test_size=0.2, shuffle=False)

# Obtención y ploteo de exactitudes
exactitudes_depth_tree = exactitudes_depth_tree(X_train, X_test, Y_train, Y_test)
exactitud_depth_tree_plot(exactitudes_depth_tree)

#%% c) ------------------------------------------------------------------------
#&& - - - - - - - - - - - - - Carga de información - - - - - - - - - - - - - - 

# Separamos en Folds y para cada uno vairamos hiperparámetros
exactitudes_foldings = k_folding (5, X_dev, Y_dev)

#%% - - - - - - - - - - - - - Ploteo de información - - - - - - - - - - - - - - 

exactitudes_promedios_train = sql^ """
                             SELECT Criterio, 
                                    Hiperparametro, 
                                    Iteracion, 
                                    AVG (Exactitud) AS AVG_Exactitud_Train
                             FROM exactitudes_foldings AS ef
                             WHERE ef."Train/Test" = 'Train'
                             GROUP BY Criterio, 
                                      Hiperparametro, 
                                      Iteracion
                             ORDER BY Hiperparametro, Iteracion, Criterio
                             """
                             
exactitudes_promedios_test = sql^ """
                             SELECT Criterio, 
                                    Hiperparametro, 
                                    Iteracion, 
                                    AVG (Exactitud) AS AVG_Exactitud_Test
                             FROM exactitudes_foldings AS ef
                             WHERE ef."Train/Test" = 'Test'
                             GROUP BY Criterio, 
                                      Hiperparametro, 
                                      Iteracion
                             ORDER BY Hiperparametro, Iteracion, Criterio
                             """

# Exactitudes promedio de los Folds variando los hiperparámetros
plot_hiperparametros (exactitudes_promedios_train, exactitudes_promedios_test)

#%% d) ========================================================================
#&& - - - - - - - - - - - - - Train Desarrollo - - - - - - - - - - - - - - - - 

# Modelo final
model_tree = DecisionTreeClassifier ( criterion='entropy', 
                                      max_depth = 10,
                                      max_features = 50,
                                      min_samples_split = 2 )

# Entrenamos modelo final
model_tree.fit(X_dev, Y_dev)

#%% - - - - - - - - - - - - - - - Test Held Out - - - - - - - - - - - - - - - -

# Predecimos el Held-Out
prediction_tree = model_tree.predict(X_held)

#%% - - - - - - - - - - - - - - - - Performance - - - - - - - - - - - - - - - -

clases = np.arange(0,10,1)    
plot_clases(pd.DataFrame(Y_held))

# Accuracy
exactitud = round(metrics.accuracy_score(Y_held, prediction_tree), 2)
print(exactitud)

# Confusion Matrix
confusion = metrics.confusion_matrix(Y_held, prediction_tree, labels = clases)
print(confusion)

# Confusion Matrix Plot
disp = metrics.ConfusionMatrixDisplay(confusion, display_labels = clases )
disp.plot()
plt.show()

#%% - - - - - - - - - - - - - - - Test Desarrollo  - - - - - - - - - - - - - -  

# Predecimos el desarrollo (Train)
prediction_tree_dev = model_tree.predict(X_dev)

#%% - - - - - - - - - - - - - - - - Performance - - - - - - - - - - - - - - - -

# Accuracy
exactitud_dev = round(metrics.accuracy_score(Y_dev, prediction_tree_dev), 2)
print(exactitud_dev)

# Confusion Matrix
confusion_dev = metrics.confusion_matrix(Y_dev, prediction_tree_dev, labels = clases)
print(confusion_dev)





