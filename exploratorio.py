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

#%% Funciones Análisis Exploratorio ===========================================
#&& a) ========================================================================
def std_data (data):
    std_data = np.std(data.iloc[:,2:], axis=0)

    return std_data

def plot_std_data (data):
    std_data = np.std(data.iloc[:,2:], axis=0)
    img = np.array(std_data, dtype = float).reshape((28,28))
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='pink')
    ax.set_xticks([])
    ax.set_yticks([])

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
            
    plt.subplots_adjust(hspace=0)
    
def plot_std_1y4_1y0 (data):
    
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
        
        sns.violinplot(x = std, 
                       ax=axs[row][1],
                       color= 'orange',
                       orient= 'h')
        axs[row][1].set_xlabel('Desviaciones estándar')
        axs[row][1].set_xlabel('Desviaciones estándar')
        axs[row][1].set_xticks(np.arange(-25,151,25))
        axs[row][1].spines['top'].set_visible(False)
        axs[row][1].spines['right'].set_visible(False)
        axs[row][1].spines['bottom'].set_visible(True)
        axs[row][1].spines['left'].set_visible(False)
        
    plt.subplots_adjust(wspace=0)

#%% b) ========================================================================

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