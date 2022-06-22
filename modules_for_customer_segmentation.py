# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:52:11 2022

@author: Tuf
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Sequential 
from tensorflow.keras.layers import BatchNormalization, Dense,Dropout 
from tensorflow.keras.activations import relu,softmax



class EDA():
    def __init__(self):
        pass
    
    def plot_cat(self,df,cat_column):
        for cat in cat_column:
            plt.figure()
            sns.countplot(df[cat],hue=df['term_deposit_subscribed'])
            plt.show()    
            
    def plot_con(self,df,con_column):
        for con in con_column:
            plt.figure()
            sns.distplot(df[con])
            plt.show()      
                    
        
class ModelCreation():
    def __init__(self):
        pass
    
    def sequential_layer(self,nb_features,num_node=128,drop_rate=0.2,output_node=2):
        model=Sequential()
        model.add(Input(shape=(nb_features))) # specific the input(no of columns)
        model.add(Dense(num_node,activation='relu',name='HiddenLayer1'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(num_node,activation='relu',name='HiddenLayer2'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,activation='softmax',name='OutputLayer'))
        model.summary()
        
        return model

class model_evaluation:
        def plot_graph(self,hist):
            plt.figure()
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.legend(['training loss', 'validation loss'])
            plt.show()
            
            plt.figure()
            plt.plot(hist.history['acc'])
            plt.plot(hist.history['val_acc'])
            plt.legend(['training acc', 'validation acc'])
            plt.show()