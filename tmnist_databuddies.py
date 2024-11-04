# -*- coding: utf-8 -*-
"""
@author: DataBuddies

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% ===========================================================================

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

del (data_1y3, data_3y8)

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

#%% Clasificación binaria =====================================================
#&& a) ========================================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from inline_sql import sql
import random

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
    
    exactitudes = pd.DataFrame(columns=[['num_atributos', 'num_vecinos', 'exactitud']])
    
    for N in n_atributos:
        for k in k_vecinos:
            
            modelo_0y1 = KNeighborsClassifier(n_neighbors=k)
            atributos = X_train.sample(n=N, axis=1).columns
            modelo_0y1.fit(X_train[atributos], Y_train)

            Y_pred = modelo_0y1.predict(X_test[atributos])
            exactitud = round(metrics.accuracy_score(Y_test, Y_pred), 2)
            
            exactitudes.loc[len(exactitudes)] = [N, k, exactitud]
            
    return exactitudes

n_atributos = np.arange(3,16,1)
k_vecinos = np.arange(3,11,1)
exactitudes = dataExactitudes(n_atributos, k_vecinos, X_train, X_test, Y_train, Y_test)

fig, ax = plt.subplots()


# Crear el bubble chart
plt.figure(figsize=(13, 8))
scatter = plt.scatter(
    exactitudes['num_vecinos'],
    exactitudes['num_atributos'],
    s=exactitudes['exactitud'] * 1000,  # Tamaño de las burbujas
    c=exactitudes['exactitud'],           # Color basado en la exactitud
    cmap='viridis',              # Mapa de colores
    alpha=0.6,
    edgecolors='w'
)

# Añadir una barra de color
plt.colorbar(scatter, label='Exactitud')

# Etiquetas y título
plt.title('Bubble Chart de Vecinos vs Atributos')
plt.xlabel('Cantidad de Vecinos')
plt.ylabel('Cantidad de Atributos')
plt.grid(True)
plt.show()

