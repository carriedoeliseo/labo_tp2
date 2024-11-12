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

#%% Funciones Clasificacion Multiclase ========================================
#&& a) ========================================================================

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

#%% b) ========================================================================

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
    
#%% c) ========================================================================

def exactitudes_hiperparametros (X_train, X_test, Y_train, Y_test):
    
    maxAtributos = np.arange(1, 82, 10)
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
             ax[i].set_xticks(np.arange(1, 82, 10))
             
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
    
#%% d) ========================================================================

def plot_clases (Y):
    
    count_clases = sql^ """
                        SELECT labels, COUNT(*) AS cantidad
                        FROM Y
                        GROUP BY labels
                        """
    
    fig, ax = plt.subplots(figsize=(6,6))
    bars = ax.bar( data=count_clases, 
                   x='labels', 
                   height='cantidad', 
                   color = 'orange',
                   alpha = 0.8,
                   edgecolor = 'black')
    ax.bar_label(bars, count_clases['cantidad'])
    ax.set_ylabel('Cantidad de muestras')
    ax.set_xlabel('Clase')
    ax.set_xticks(np.arange(0,10,1))