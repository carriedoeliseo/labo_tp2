# ================= LABORATORIO DE DATOS: TRABAJO PRÁCTICO 2 ==================
# ============================ GRUPO E INTEGRANTES ============================
'''
 Data Buddies:
    -Eliseo Carriedo  (L.U.: 392/23)
    -Lila Fage (L.U.: 235/24)
    -Julian Laurido (L.U.: 1097/23 )
'''
#%% IMPORTS ===================================================================

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

#%% Funciones Clasificacion Binaria ===========================================
#&& c) ========================================================================

def exactitud_0y1_por_atributos (n_atributos, X_train, X_test, Y_train, Y_test):
    
    exactitudes_test = []
    exactitudes_train = []
    
    for N in n_atributos:
        
        modelo_0y1 = KNeighborsClassifier(n_neighbors=5)
        atributos = X_train.sample(n=N, axis=1).columns
        modelo_0y1.fit(X_train[atributos], Y_train)

        Y_test_pred = modelo_0y1.predict(X_test[atributos])
        exactitudes_test.append(round(metrics.accuracy_score(Y_test, Y_test_pred), 2))
        
        Y_train_pred = modelo_0y1.predict(X_train[atributos])
        exactitudes_train.append(round(metrics.accuracy_score(Y_train, Y_train_pred), 2))
        
    iteraciones = np.arange(1, len(exactitudes_test)+1, 1)
    fig, ax = plt.subplots()
    
    ax.plot(iteraciones, 
            exactitudes_test, 
            '-o', 
            color = 'purple', 
            label='Test')
    
    ax.plot(iteraciones, 
            exactitudes_train, 
            '-o', label='Train', 
            linestyle='--', 
            alpha=0.5, 
            color='gray')
    
    ax.legend(title='Predict', loc='lower right')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Exactitud')
    ax.set_xticks([3,6,9,12,15,18])
    
#%% c) ========================================================================

def dataExactitudes (n_atributos, k_vecinos, X_train, X_test, Y_train, Y_test):
    
    exactitudes = pd.DataFrame(columns=['num_atributos', 'num_vecinos', 'exactitud'])
    
    for N in n_atributos:
        for k in k_vecinos:
            
            modelo_0y1 = KNeighborsClassifier(n_neighbors=k)
            atributos = X_train.sample(n=N, axis=1).columns
            modelo_0y1.fit(X_train[atributos], Y_train)

            Y_pred = modelo_0y1.predict(X_test[atributos])
            
            exactitud = round(metrics.accuracy_score(Y_test, Y_pred), 2)
            exactitudes.loc[len(exactitudes)] = [N, k, exactitud]
            
    return exactitudes

def exactitudes_plot (exactitudes):
    
    fig, ax = plt.subplots(figsize=(14,11))
    
    scatter = ax.scatter(exactitudes['num_vecinos'],
                         exactitudes['num_atributos'],
                         s=exactitudes['exactitud']*2000,
                         c=exactitudes['exactitud'],
                         cmap='viridis',
                         alpha=0.7,
                         marker='o')
    
    fig.colorbar(scatter, label='Exactitud')
    ax.set_xlabel('Cantidad de Vecinos', size=15)
    ax.set_ylabel('Cantidad de Atributos', size=15)
    ax.set_xticks(np.arange(3,16,1))
    ax.set_yticks(np.arange(3,16,1))
    ax.grid(True)