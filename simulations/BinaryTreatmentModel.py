from sklearn.metrics import mean_squared_error

import warnings

import sys
sys.path.append('../../')

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

import tensorflow as tf

from keras.layers import Input, Dense, Add, multiply, Activation
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate

from keras import regularizers

from keras.backend import zeros, ones
import keras.backend as K
from keras.layers.core import Lambda

import numpy as np
import pandas as pd

########################
# Fix the architecture #
########################

def make_shared_net(input_shape, activations, hidden_layers,
                    dropout_rate, l2, constrain_norm=False):
    
    if isinstance(activations, str):
        activations = [activations] * len(hidden_layers)
    
    m = Sequential(name='h_hat')
    first_layer = True
    for h, a in zip(hidden_layers, activations):
        if l2 > 0.:
            w_reg = regularizers.l2(l2)
        else:
            w_reg = None
        const = maxnorm(2) if constrain_norm else  None
        
        if first_layer:
            m.add(Dense(h, input_shape=(input_shape,), activation=a,
                        kernel_regularizer=w_reg, kernel_constraint=const))
            first_layer = False
        else:
            m.add(Dense(h, activation=a, kernel_regularizer=w_reg, 
                        kernel_constraint=const))
        m.add(Activation(a))
        
    #Add final dense layer
    
    m.add(Dense(1))
    
    return m

def add0(features):
    tf_constant = K.constant(np.array([[0]]))
    batch_size = K.shape(features)[0]
    tiled_constant = K.tile(tf_constant, (batch_size, 1))
    return Concatenate(axis=1)([features, tiled_constant])

def add1(features):
    tf_constant = K.constant(np.array([[1]]))
    batch_size = K.shape(features)[0]
    tiled_constant = K.tile(tf_constant, (batch_size, 1))
    return Concatenate(axis=1)([features, tiled_constant])