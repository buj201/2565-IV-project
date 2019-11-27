import sys
sys.path.append('../DeepIV/experiments/')

import data_generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook',font_scale=1.3)

from keras.utils.vis_utils import plot_model,model_to_dot

from keras.backend import zeros, ones
import keras.backend as K
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers

from keras.layers import Input, Dense, Add, Multiply, Activation
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate

from keras import regularizers

from keras.backend import zeros, ones
import keras.backend as K
from keras.layers.core import Lambda

from ..BinaryTreatmentModel import *

from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import pydot

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

hidden = [128, 64, 32]
act = "relu"    
dropout_rate = 0.5
l2 = 0.001

np.random.seed(1342)

x = np.zeros(shape=(10,10))
pi0 = np.zeros(shape=(10,1))
pi1 = np.zeros(shape=(10,1))

_VALID_OP_NAME_REGEX='.*'

features = Input(shape=(x.shape[1],), name="Features")
pi0_input = Input(shape=(pi0.shape[1],), name="pi0")
pi1_input = Input(shape=(pi1.shape[1],), name="pi1")

response_input0 = Lambda(add0, name='Add0')(features)
response_input1 = Lambda(add1, name='Add1')(features)

shared = make_shared_net(x.shape[1]+1, act, hidden, dropout_rate=dropout_rate, l2=l2)

h0 = shared(response_input0)
h1 = shared(response_input1)

weighted_yhat = Add(name='weighted_yhat')([
    Multiply(name='mult0')([pi0_input, h0]),
    Multiply(name='mult1')([pi1_input, h1]),
])

response_model = Model(inputs=[features, pi0_input, pi1_input],
                       outputs=weighted_yhat)
response_model.compile('adam', loss='mse')

model_to_dot(response_model).write('output/figures/dot_file.dot')

new_names = {
    'Features: InputLayer': "X=x",
    'Add0: Lambda': 'Concatenate (x, T=0)',
    'Add1: Lambda': 'Concatenate (x, T=1)',
    'pi0: InputLayer': 'p(T=0\|X=x,Z=z) (from stage 1 DNN)',
    'pi1: InputLayer': 'p(T=1\|X=x,Z=z) (from stage 1 DNN)',
    'h_hat: Sequential': 'DNN h(x,t)',
    'mult1: Multiply': 'p(T=1\|X=x)h(x,t=1)',
    'mult0: Multiply': 'p(T=0\|X=x)h(x,t=0)',
    'weighted_yhat: Add': 'p(T=0\|X=x)h(x,t=0) + p(T=1\|X=x)h(x,t=1)',
}

for old, new in new_names.items():
    replace('output/figures/dot_file.dot',old,new)
    
(graph,) = pydot.graph_from_dot_file('output/figures/dot_file.dot')
graph.write_png('output/figures/binary_DeepIV_model.png')