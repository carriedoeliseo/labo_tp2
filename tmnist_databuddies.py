"""
# -*- coding: utf-8 -*-
"""

# ================= LABORATORIO DE DATOS: TRABAJO PRÁCTICO 2 ==================
# ============================ GRUPO E INTEGRANTES ============================

# Data Buddies:
#    -Eliseo Carriedo  (L.U.: 392/23)
#    -Lila Fage (L.U.: 235/24)
#    -Julian Laurido (L.U.: 1097/23 )

# ================================ DESCRIPCION ================================

#
#
#

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

#%% CARGA DE DATOS ============================================================

carpeta = './'
data = pd.read_csv(carpeta+'TMNIST_Data.csv')
columnas_num = data.select_dtypes(include=[np.int64]).columns[1:]
data[columnas_num] = data[columnas_num].astype(np.int32)

del(columnas_num)

#%% ANALISIS EXPLORATORIO =====================================================
#&& a) ========================================================================

def std_data (data):
    std_data = np.std(data.iloc[:,2:], axis=0)
    img = np.array(std_data, dtype = float).reshape((28,28))
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='pink')
    ax.set_xticks([])
    ax.set_yticks([])
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
            axs[row, col].imshow(img, cmap='pink')
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            numero += 1

# Promedio intensidad de cada pixel por número
plot_media_numero(data)

def plot_std_1y3_3y8 (data):
    
    data_1y3 = data[(data['labels'] == 1) | (data['labels'] == 4)].iloc[:,2:]
    data_3y8 = data[(data['labels'] == 1) | (data['labels'] == 0)].iloc[:,2:]
    
    datas = [data_1y3, data_3y8]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    for row in range (2):
        std = np.std(datas[row], axis = 0)
        img = np.array(std).reshape((28,28))
        axs[row][0].imshow(img, cmap='pink')
        
        axs[row][0].set_xticks([])
        axs[row][0].set_yticks([])
        axs[row][0].set_title('('+str(row +1)+')', loc='left')
        
        sns.histplot(x=std, ax=axs[row][1], bins=20, color = 'orange')
        axs[row][1].set_xlabel('Desviaciones estándar')
        axs[row][1].set_ylabel('')
        counts, _ = np.histogram(std, bins=20)
        max_count = counts.max()
        axs[row][1].axhline(y=max_count, color='black', linestyle='--', linewidth=0.5)
        axs[row][1].set_xticks(np.arange(0, max(std) +1, 10))
        axs[row][1].set_yticks([max_count])
        
    plt.subplots_adjust(wspace=0)
            
# Variabilidad de intensidad entre 1 y 3 y entre 3 y 8
plot_std_1y3_3y8(data)

#%% c) ========================================================================
                       
def plot_numeros (data):
    nums = [0,1,4,7]
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(13,6))
    
    for col in range (4):
        data_col = data[data['labels'] == nums[col]].iloc[:,2:]
        datas = [np.mean(data_col, axis = 0), np.std(data_col, axis = 0)]
        
        for row in range (2):
            img = np.array(datas[row]).reshape((28,28))
            axs[row, col].imshow(img, cmap='pink')
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
   
    plt.subplots_adjust(wspace=0, hspace=0.1)
    
# Variabilidad y promedio de la clase 0
plot_numeros (data)

#%% d) ========================================================================

"""
La exploración se complica debido a la necesidad de herramientas de 
visualizacion.
"""

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
    
n_atributos = [3]*15
exactitud_0y1_por_atributos (n_atributos, X_train, X_test, Y_train, Y_test)

n_atributos = np.arange(3,19,1)
exactitud_0y1_por_atributos (n_atributos, X_train, X_test, Y_train, Y_test)

#%% d) ========================================================================

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

    
n_atributos = np.arange(3,16,1)
k_vecinos = np.arange(3,16,1)
exactitudes = dataExactitudes(n_atributos, k_vecinos, X_train, X_test, Y_train, Y_test)

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
    ax.set_xlabel('Cantidad de Vecinos')
    ax.set_ylabel('Cantidad de Atributos')
    ax.set_xticks(np.arange(3,16,1))
    ax.set_yticks(np.arange(3,16,1))
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
    
    exactitudes_tree = pd.DataFrame(columns=['Train/Test', 'Profundidad', 'Exactitud'])
    
    for i in range(1,11):
        
        model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        model_tree.fit(X_train, Y_train)
        
        # ------------------------------- TEST --------------------------------
        prediction_tree = model_tree.predict(X_test)
        exactitud = round(metrics.accuracy_score(Y_test, prediction_tree), 2)
        exactitudes_tree.loc[len(exactitudes_tree)] = ['Test', i, exactitud]
        
        # ------------------------------ TRAIN --------------------------------
        prediction_tree = model_tree.predict(X_train)
        exactitud = round(metrics.accuracy_score(Y_train, prediction_tree), 2)
        exactitudes_tree.loc[len(exactitudes_tree)] = ['Train', i, exactitud]
        
    return exactitudes_tree
    
exactitudes_depth_tree = exactitudes_depth_tree(X_train, X_test, Y_train, Y_test)

def exactitud_depth_tree_plot (exactitudes_tree):
    
    exactitudes_test = exactitudes_tree[exactitudes_tree['Train/Test'] == 'Test']
    exactitudes_train = exactitudes_tree[exactitudes_tree['Train/Test'] == 'Train']
    
    fig, ax = plt.subplots()
    
    ax.plot(exactitudes_test['Profundidad'], 
            exactitudes_test['Exactitud'], 
            '-o', 
            color = 'purple',
            label='Test')
    
    ax.plot(exactitudes_train['Profundidad'], 
            exactitudes_train['Exactitud'], 
            '-o', linestyle='--', 
            alpha=0.5, 
            color='gray', 
            label='Train')
    
    ax.set_xlabel('Profundidades')
    ax.set_ylabel('Exactitud')
    ax.legend(title='Predict', loc='lower right')
    ax.set_xticks(np.arange(1,11,1))
    
exactitud_depth_tree_plot(exactitudes_depth_tree)

#%% c) ========================================================================
#&& Carga de información ======================================================

def exactitudes_hiperparametros (X_train, X_test, Y_train, Y_test):
    
    maxAtributos = np.arange(5, 11, 1)
    profundidades = np.arange(5, 11, 1)
    minEjemplares = [2,4,8,16,32]

    exactitudes = pd.DataFrame(columns=['Train/Test','Criterio', 'Hiperparametro', 'Iteracion', 'Exactitud'])
    hiperparametros = ['MaxFeatures', 'Profundidad', 'minSamples']
    
    for h in hiperparametros:
        
        if h == 'MaxFeatures':
                for i in maxAtributos:
                    
                    # ------------------------ ENTROPY ------------------------
                    model_tree_entr = DecisionTreeClassifier(criterion='entropy', max_features=i)
                    model_tree_entr.fit(X_train, Y_train)
                    
                    # Accuracy con test
                    prediction_tree_entr = model_tree_entr.predict(X_test)
                    exactitud = round(metrics.accuracy_score(Y_test, prediction_tree_entr), 2)
                    exactitudes.loc[len(exactitudes)] = ['Test', 'entropy', h, i, exactitud]
                    
                    # Accuracy con train
                    prediction_tree_entr = model_tree_entr.predict(X_train)
                    exactitud = round(metrics.accuracy_score(Y_train, prediction_tree_entr), 2)
                    exactitudes.loc[len(exactitudes)] = ['Train', 'entropy', h, i, exactitud]
                    
                    # ------------------------- GINI --------------------------
                    model_tree_gini = DecisionTreeClassifier(criterion='gini', max_features=i)
                    model_tree_gini.fit(X_train, Y_train)
                    
                    # Accuracy con test
                    prediction_tree_gini = model_tree_gini.predict(X_test)
                    exactitud = round(metrics.accuracy_score(Y_test, prediction_tree_gini), 2)
                    exactitudes.loc[len(exactitudes)] = ['Test', 'gini', h, i, exactitud]
                    
                    # Accuracy con train
                    prediction_tree_gini = model_tree_gini.predict(X_train)
                    exactitud = round(metrics.accuracy_score(Y_train, prediction_tree_gini), 2)
                    exactitudes.loc[len(exactitudes)] = ['Train', 'gini', h, i, exactitud]
                    
        elif h == 'Profundidad':
                for k in profundidades:
                    
                    # ------------------------ ENTROPY ------------------------
                    model_tree_entr = DecisionTreeClassifier(criterion='entropy', max_depth=k)
                    model_tree_entr.fit(X_train, Y_train)
                    
                    # Accuracy con test
                    prediction_tree_entr = model_tree_entr.predict(X_test)
                    exactitud = round(metrics.accuracy_score(Y_test, prediction_tree_entr), 2)
                    exactitudes.loc[len(exactitudes)] = ['Test', 'entropy', h, k, exactitud]
                    
                    # Accuracy con train
                    prediction_tree_entr = model_tree_entr.predict(X_train)
                    exactitud = round(metrics.accuracy_score(Y_train, prediction_tree_entr), 2)
                    exactitudes.loc[len(exactitudes)] = ['Train', 'entropy', h, k, exactitud]
                    
                    # ------------------------- GINI --------------------------
                    model_tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=k)
                    model_tree_gini.fit(X_train, Y_train)
                    
                    # Accuracy con test
                    prediction_tree_gini = model_tree_gini.predict(X_test)
                    exactitud = round(metrics.accuracy_score(Y_test, prediction_tree_gini), 2)
                    exactitudes.loc[len(exactitudes)] = ['Test', 'gini', h, k, exactitud]
                    
                    # Accuracy con train
                    prediction_tree_gini = model_tree_gini.predict(X_train)
                    exactitud = round(metrics.accuracy_score(Y_train, prediction_tree_gini), 2)
                    exactitudes.loc[len(exactitudes)] = ['Train', 'gini', h, k, exactitud]
                    
        elif h == 'minSamples':  
                for l in minEjemplares:
                    
                    # ------------------------ ENTROPY ------------------------
                    model_tree_entr = DecisionTreeClassifier(criterion='entropy', min_samples_split=l)
                    model_tree_entr.fit(X_train, Y_train)
                    
                    # Accuracy con test
                    prediction_tree_entr = model_tree_entr.predict(X_test)
                    exactitud = round(metrics.accuracy_score(Y_test, prediction_tree_entr), 2)
                    exactitudes.loc[len(exactitudes)] = ['Test', 'entropy', h, l, exactitud]
                    
                    # Accuracy con train
                    prediction_tree_entr = model_tree_entr.predict(X_train)
                    exactitud = round(metrics.accuracy_score(Y_train, prediction_tree_entr), 2)
                    exactitudes.loc[len(exactitudes)] = ['Train', 'entropy', h, l, exactitud]
                    
                    # ------------------------- GINI --------------------------
                    model_tree_gini = DecisionTreeClassifier(criterion='gini', min_samples_split=l)
                    model_tree_gini.fit(X_train, Y_train)
                    
                    # Accuracy con test
                    prediction_tree_gini = model_tree_gini.predict(X_test)
                    exactitud = round(metrics.accuracy_score(Y_test, prediction_tree_gini), 2)
                    exactitudes.loc[len(exactitudes)] = ['Test', 'gini', h, l, exactitud]
                    
                    # Accuracy con train
                    prediction_tree_gini = model_tree_gini.predict(X_train)
                    exactitud = round(metrics.accuracy_score(Y_train, prediction_tree_gini), 2)
                    exactitudes.loc[len(exactitudes)] = ['Train', 'gini', h, l, exactitud]
                    
    return exactitudes

def k_folding (nsplits, X_dev, Y_dev):
    
    resultados = pd.DataFrame(columns=['Fold', 'Train/Test', 'Criterio', 'Hiperparametro', 'Iteracion', 'Exactitud'])
    
    kf = KFold(n_splits=nsplits)
    for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
    
        kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
        kf_Y_train, kf_Y_test = Y_dev.iloc[train_index], Y_dev.iloc[test_index]
        
        exactitudes_fold = exactitudes_hiperparametros(kf_X_train, kf_X_test, kf_Y_train, kf_Y_test)
        exactitudes_fold['Fold'] = i
        resultados = pd.concat([resultados, exactitudes_fold], axis=0)
        
    return resultados

exactitudes_foldings = k_folding (5, X_dev, Y_dev)

#%% Ploteo de información =====================================================

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
                             
def plot_hiperparametros (exactitudes_promedios_train, exactitudes_promedios_test):
    
    hiperparametros = ['MaxFeatures', 'Profundidad', 'minSamples']
    fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(10,15))
    i = 0
    for h in hiperparametros:
        
         exactitudes_entr_train = sql^ """
                                       SELECT Iteracion, AVG_Exactitud_Train
                                       FROM exactitudes_promedios_train
                                       WHERE Criterio = 'entropy' AND
                                             Hiperparametro = $h
                                       """
         exactitudes_gini_train = sql^ """
                                       SELECT Iteracion, AVG_Exactitud_Train
                                       FROM exactitudes_promedios_train
                                       WHERE Criterio = 'gini' AND
                                             Hiperparametro = $h
                                       """
                                       
         exactitudes_entr_test = sql^ """
                                       SELECT Iteracion, AVG_Exactitud_Test
                                       FROM exactitudes_promedios_test
                                       WHERE Criterio = 'entropy' AND
                                             Hiperparametro = $h
                                       """
         exactitudes_gini_test = sql^ """
                                       SELECT Iteracion, AVG_Exactitud_Test
                                       FROM exactitudes_promedios_test
                                       WHERE Criterio = 'gini' AND
                                             Hiperparametro = $h
                                       """
                                 
         ax[i].plot(exactitudes_entr_train['Iteracion'], 
                    exactitudes_entr_train['AVG_Exactitud_Train'],
                    '-o',
                    linestyle='--',
                    label='Entropy Train',
                    color='thistle',
                    alpha = 0.5)
         
         ax[i].plot(exactitudes_gini_train['Iteracion'], 
                    exactitudes_gini_train['AVG_Exactitud_Train'],
                    'o-',
                    linestyle='--', 
                    label='Gini Train',
                    color='moccasin',
                    alpha = 0.5)
         
         ax[i].plot(exactitudes_entr_test['Iteracion'], 
                    exactitudes_entr_test['AVG_Exactitud_Test'],
                    'o-', 
                    label='Entropy Test',
                    color='purple',
                    alpha = 0.5)
         
         ax[i].plot(exactitudes_gini_test['Iteracion'], 
                    exactitudes_gini_test['AVG_Exactitud_Test'],
                    'o-',
                    label='Gini Test',
                    color='darkorange',
                    alpha = 0.5)
         
         if h == 'MaxFeatures':
             ax[i].set_title('(1)', loc='right')
             ax[i].set_xlabel('Cantidad de Atributos')
             ax[i].set_ylabel('Exactitud')
             ax[i].set_xticks(np.arange(5, 11, 1))
             
         elif h == 'Profundidad':
             ax[i].set_title('(2)', loc='right')
             ax[i].set_xlabel('Profundidad del árbol')
             ax[i].set_ylabel('Exactitud')
             ax[i].set_xticks(np.arange(5, 11, 1))
             
         elif h == 'minSamples':
             ax[i].set_title('(3)', loc='right')
             ax[i].set_xlabel('Minimo de Ejemplares')
             ax[i].set_ylabel('Exactitud')
             ax[i].set_xticks([2,4,8,16,32])
         
         fig.legend(['Entropy Train', 'Gini Train', 'Entropy Test', 'Gini Test'],
                    bbox_to_anchor=(0.275, 1.07))
         plt.subplots_adjust(top=1)
         i+=1
         
    plt.show()
    
plot_hiperparametros (exactitudes_promedios_train, exactitudes_promedios_test)

