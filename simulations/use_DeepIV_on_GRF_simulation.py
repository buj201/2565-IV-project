##########################################################################
# Adapted from jhartford/DeepIV/experiments/demand_simulation.py
##########################################################################

from sklearn.metrics import mean_squared_error

import warnings

import sys
sys.path.append('../')

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.merge import Concatenate

import numpy as np
import pandas as pd

########################
# Load the dataset 
########################

from joblib import load
all_data = load('output/temp/data_for_DeepIV.pkl')
locals().update(all_data)

x, z, t, y = X_train, Z_train, W_train, Y_train


print("Data shapes:\n\
Features:{x},\n\
Instruments:{z},\n\
Treament:{t},\n\
Response:{y}".format(**{'x':x.shape, 'z':z.shape,
                        't':t.shape, 'y':y.shape}))


########################
# Set model parameters
########################

n = x.shape[0]

dropout_rate = min(1000./(1000. + n), 0.5)
epochs = int(1500000./float(n)) # heuristic to select number of epochs
l2 = 0.0001
batch_size = 100
hidden = [128, 64, 32]
act = "relu"

# Build and fit treatment model
instruments = Input(shape=(z.shape[1],), name="instruments")
features = Input(shape=(x.shape[1],), name="features")
treatment_input = Concatenate(axis=1)([instruments, features])

est_treat = architectures.feed_forward_net(treatment_input,
                                           output=Dense(1, activation='sigmoid'),
                                           hidden_layers=hidden,
                                           dropout_rate=dropout_rate, l2=l2,
                                           activations=act)

treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
treatment_model.compile('adam',
                        loss="binary_crossentropy")

treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

# Build and fit response model

treatment = Input(shape=(t.shape[1],), name="treatment")
response_input = Concatenate(axis=1)([features, treatment])

est_response = architectures.feed_forward_net(response_input,
                                              output=Dense(1),
                                              activations=act,
                                              hidden_layers=hidden,
                                              l2=l2,
                                              dropout_rate=dropout_rate)

response_model = Response(treatment=treatment_model,
                          inputs=[features, treatment],
                          outputs=est_response)
response_model.compile('adam', loss='mse')
response_model.fit([z, x], y, epochs=epochs, verbose=1,
                   batch_size=batch_size, samples_per_batch=2)

####################################
# The GRF simulation has a binary
# treatment W -- so the treatment effect
# is the difference h(x,w=1) - h(x,w=0)
####################################

DeepIV_tau_test = response_model.predict([X_test,np.ones_like(W_test)]) -\
                  response_model.predict([X_test,np.zeros_like(W_test)])
DeepIV_MSE = mean_squared_error(y_true = tau_test,
                                y_pred = DeepIV_tau_test)


params = load('output/temp/params.pkl')
params['DeepIV_MSE'] = DeepIV_MSE

pd.Series(params).to_frame().T.set_index('seed')\
  .to_csv('output/MSEs/MSE_test.csv')