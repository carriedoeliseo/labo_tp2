"""
# -*- coding: utf-8 -*-
"""

################# LABORATORIO DE DATOS: TRABAJO PRÁCTICO 2 ####################


########################### GRUPO E INTEGRANTES ###############################

# Data Buddies:
#    -Eliseo Carriedo  (L.U.: 392/23)
#    -Lila Fage (L.U.: 235/24)
#    -Julian Laurido (L.U.: 1097/23 )

############################### DESCRIPCION ###################################

#
#
#

#%% ===========================================================================
########################## SECCION DE IMPORTS #################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from inline_sql import sql
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#%% ===========================================================================
########################### CARGA DE DATOS ####################################
carpeta = './'
data = pd.read_csv(carpeta+'TMNIST_Data.csv')

#%% ANALISIS EXPLORATORIO =====================================================
#&& a) ========================================================================

def std_data (data):
    std_data = np.std(data.iloc[:,2:], axis=0)
    img = np.array(std_data, dtype = float).reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.show()
    return std_data
    
no_importantes = std_data(data)[std_data(data) == 0].index

#%% b) ========================================================================

def plot_media_numero (data):
    
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15,6))
    numero = 0
    
    for row in range (2):
        for col in range (5):
            data_numero = data[data['labels'] == numero].iloc[:,2:]
            data_media_numero = np.mean(data_numero, axis = 0)
            img = np.array(data_media_numero, dtype = float).reshape((28,28))
            axs[row, col].imshow(img, cmap='gray')
            numero += 1

# Promedio intensidad de cada pixel por número
plot_media_numero(data)

data_1y3 = data[(data['labels'] == 1) | (data['labels'] == 3)].iloc[:,2:]
data_3y8 = data[(data['labels'] == 3) | (data['labels'] == 8)].iloc[:,2:]

def plot_std_1y3_3y8 (data_1y3, data_3y8):
    datas = [data_1y3, data_3y8]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13,6))
    for col in range (2):
        std = np.std(datas[col], axis = 0)
        img = np.array(std).reshape((28,28))
        axs[col].imshow(img, cmap='gray')
            
# Variabilidad de intensidad entre 1 y 3 y entre 3 y 8
plot_std_1y3_3y8(data_1y3, data_3y8)

#%% c) ========================================================================
                       
def plot_clase_0 (data):
    data_0 = data[data['labels'] == 0].iloc[:,2:]
    datas = [np.mean(data_0, axis = 0), np.std(data_0, axis = 0)]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13,6))
    for col in range (2):
        img = np.array(datas[col]).reshape((28,28))
        axs[col].imshow(img, cmap='gray')
        
# Variabilidad y promedio de la clase 0
plot_clase_0 (data)

#%% d) ========================================================================

"""
La exploración se complica debido a la necesidad de herramientas de 
visualizacion.
"""

#%% Borrado ===================================================================

del (data_1y3, data_3y8)

#%% Clasificación binaria =====================================================
#&& a) ========================================================================

data_0y1 = data[(data['labels'] == 0) | (data['labels'] == 1)].iloc[:,1:].reset_index(drop=True)

# Está re balanceado
count_0y1 = sql^ """
                 SELECT labels, COUNT(*) AS cantidad
                 FROM data_0y1
                 GROUP BY labels
                 """

#%% b) ========================================================================

std_0y1 = np.std(data_0y1.iloc[:,1:], axis = 0)
no_importantes_0y1 = std_0y1[std_0y1 == 0].index
importantes_0y1 = data_0y1.drop(no_importantes_0y1, axis = 1)

X = importantes_0y1.iloc[:,1:]
Y = importantes_0y1['labels']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

#%% c) ========================================================================

def exactitud_0y1_por_atributos (n_atributos, X_train, X_test, Y_train, Y_test):
    exactitudes = []
    for N in n_atributos:
        
        modelo_0y1 = KNeighborsClassifier(n_neighbors=5)
        atributos = X_train.sample(n=N, axis=1).columns
        modelo_0y1.fit(X_train[atributos], Y_train)

        Y_pred = modelo_0y1.predict(X_test[atributos])
        exactitudes.append(round(metrics.accuracy_score(Y_test, Y_pred), 2))
        
    iteraciones = np.arange(1, len(exactitudes)+1, 1)
    fig, ax = plt.subplots()
    ax.plot(iteraciones,exactitudes)

n_atributos = [3]*10
exactitud_0y1_por_atributos (n_atributos, X_train, X_test, Y_train, Y_test)

n_atributos = np.arange(3,20,1)
exactitud_0y1_por_atributos (n_atributos, X_train, X_test, Y_train, Y_test)

#%% d) ========================================================================

def dataExactitudes (n_atributos, k_vecinos, X_train, X_test, Y_train, Y_test):
    
    exactitudes = pd.DataFrame(columns=['num_atributos', 'num_vecinos', 'exactitud', 'precision_0', 'recall_0', 'precision_1', 'recall_1'])
    
    for N in n_atributos:
        for k in k_vecinos:
            
            modelo_0y1 = KNeighborsClassifier(n_neighbors=k)
            atributos = X_train.sample(n=N, axis=1).columns
            modelo_0y1.fit(X_train[atributos], Y_train)

            Y_pred = modelo_0y1.predict(X_test[atributos])
            
            exactitud = round(metrics.accuracy_score(Y_test, Y_pred), 2)
            confusion = metrics.confusion_matrix(Y_test, Y_pred, labels=[0, 1])
            
            prec0 = round((confusion[0,0])/sum(confusion[:,0]), 2)
            reca0 = round((confusion[0,0])/sum(confusion[0,:]), 2)
            prec1 = round((confusion[1,1])/sum(confusion[:,1]), 2)
            reca1 = round((confusion[1,1])/sum(confusion[1,:]), 2)
            
            exactitudes.loc[len(exactitudes)] = [N, k, exactitud, prec0, reca0, prec1, reca1]
            
    return exactitudes

    
n_atributos = np.arange(3,16,1)
k_vecinos = np.arange(3,11,1)
exactitudes = dataExactitudes(n_atributos, k_vecinos, X_train, X_test, Y_train, Y_test)

def exactitudes_plot (exactitudes):
    
    fig, ax = plt.subplots(figsize=(13, 8))
    scatter = ax.scatter(exactitudes['num_vecinos'],
                         exactitudes['num_atributos'],
                         s=exactitudes['exactitud']*1000,
                         c=exactitudes['exactitud'],
                         cmap='viridis',
                         alpha=0.6,
                         edgecolors='w')
    fig.colorbar(scatter, label='Exactitud')
    ax.set_title('Bubble Chart de Vecinos vs Atributos')
    ax.set_label('Cantidad de Vecinos')
    ax.set_label('Cantidad de Atributos')
    ax.grid(True)
    
exactitudes_plot(exactitudes)

#%% Borrado ===================================================================

del (count_0y1, data_0y1, importantes_0y1, n_atributos, no_importantes_0y1, std_0y1, X, Y,X_train, X_test, Y_train, Y_test)

#%% Clasificacion Multiclase ==================================================
#&& a) ========================================================================

data_importante = data.drop(no_importantes, axis = 1)
X = data_importante.iloc[:,2:]
Y = data_importante.iloc[:,1]
X_dev, X_held, Y_dev, Y_held = train_test_split(X, Y, test_size=0.1, shuffle=False)

#%% b) ========================================================================

X_train, X_test, Y_train, Y_test = train_test_split(X_dev, Y_dev, test_size=0.2, shuffle=False)

def exactitudes_depth_tree(X_train, X_test, Y_train, Y_test):
    
    exactitudes_tree = pd.DataFrame(columns=['Profundidad', 'Precision'])
    
    for i in range(1,11):
        
        model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        model_tree.fit(X_train, Y_train)
        prediction_tree = model_tree.predict(X_test)
        
        #A partir de profundidad 8, la accuracy se mantiene estable alrededor de [0.92-0.93]
        exactitud = round(metrics.accuracy_score(Y_test, prediction_tree), 2)
        exactitudes_tree.loc[len(exactitudes_tree)] = [i, exactitud]
        
    return exactitudes_tree
    
exactitudes_depth_tree = exactitudes_depth_tree(X_train, X_test, Y_train, Y_test)

def exactitud_depth_tree_plot (exactitudes_tree):
    
    fig, ax = plt.subplots()
    ax.plot(exactitudes_tree['Profundidad'], exactitudes_tree['Precision'])
    ax.set_title('Grafico de barras de precisiones')
    ax.set_xlabel('Profundidades')
    ax.set_ylabel('Precision del Arbol')
    ax.set_xticks(np.arange(1,11,1))
    
exactitud_depth_tree_plot(exactitudes_depth_tree)

#%% c) ========================================================================

def exactitudes_hiperparametros (X_train, X_test, Y_train, Y_test):
    
    exactitudes_tree = pd.DataFrame(columns=['Criterio', 'maxAtributos', 'Profundidad', 'minSamples','Precision'])
    maxAtributos = np.arange(2, 15, 3)
    profundidades = np.arange(1, 10, 2)
    minEjemplares = np.arange(2, 10, 2)
    
    fig, axs = plt.subplots(ncols=3)
    
    exactitudes = pd.DataFrame(columns=['MaxFeatures', 'Precision'])
    
    for i in maxAtributos:
        
        model_tree = DecisionTreeClassifier(criterion='entropy', max_features=i)
        model_tree.fit(X_train, Y_train)
        prediction_tree = model_tree.predict(X_test)
        
        exactitud = round(metrics.accuracy_score(Y_test, prediction_tree), 2)
        exactitudes.loc[len(exactitudes_tree)] = [i, exactitud]
        
        axs[0].plot(exactitudes['MaxFeatures'], exactitudes['Precision'])
        
    for i in maxAtributos:
        
        model_tree = DecisionTreeClassifier(criterion='gini', max_features=i)
        model_tree.fit(X_train, Y_train)
        prediction_tree = model_tree.predict(X_test)
        
        exactitud = round(metrics.accuracy_score(Y_test, prediction_tree), 2)
        exactitudes.loc[len(exactitudes_tree)] = [i, exactitud]
        
        axs[0].plot(exactitudes['MaxFeatures'], exactitudes['Precision'])
    
    
    

