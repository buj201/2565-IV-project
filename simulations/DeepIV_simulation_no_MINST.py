#####################
# Standard imports 
#####################

from joblib import dump
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings

import sys
sys.path.append('../')

#####################
# DeepIV imports 
#####################

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.merge import Concatenate

sys.path.append('../DeepIV/experiments')
import data_generator

#####################
# Generate data
#####################

n = 5000
dropout_rate = min(1000./(1000. + n), 0.5)
epochs = int(1500000./float(n)) # heuristic to select number of epochs
epochs = 300
batch_size = 100
images = False
seed = 6434
ypcor = 0.5

hidden = [128, 64, 32]
act = "relu"
n_components = 10
l2 = 0.001

params = {
    'n': n,
    'dropout_rate': dropout_rate,
    'epochs': epochs,
    'batch_size': batch_size,
    'images': images,
    'seed': seed,
    'hidden': hidden,
    'activation': act,
    'n_components': n_components,
    'l2': l2
}

def datafunction(n, s, images=images, test=False):
    return data_generator.demand(n=n, seed=s, ypcor=ypcor,
                                 use_images=images, test=test)

X_train, Z_train, W_train, Y_train, _ = datafunction(n, seed)
X_test, Z_test, W_test, Y_test, _ = datafunction(n, seed)
tau_test = data_generator.true_tau(X_test)

all_data = {
    'X_train': X_train,
    'Y_train': Y_train.flatten(),
    'W_train': W_train.flatten(),
    'Z_train': Z_train.flatten(),
        
    'X_test': X_test,
    'Y_test': Y_test.flatten(),
    'W_test': W_test.flatten(),
    'Z_test': Z_test.flatten(),
    'tau_test': tau_test
}

dump(all_data,'output/temp/data_for_GRF.pkl')

##########################################
# Build and fit treatment model
##########################################

x, y, t, z = X_train, Y_train, W_train, Z_train

instruments = Input(shape=(z.shape[1],), name="instruments")
features = Input(shape=(x.shape[1],), name="features")
treatment_input = Concatenate(axis=1)([instruments, features])

est_treat = architectures.feed_forward_net(treatment_input, lambda x: densities.mixture_of_gaussian_output(x, n_components),
                                           hidden_layers=hidden,
                                           dropout_rate=dropout_rate, l2=0.0001,
                                           activations=act)

treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
treatment_model.compile('adam',
                        loss="mixture_of_gaussians",
                        n_components=n_components)

treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

# Build and fit response model

treatment = Input(shape=(t.shape[1],), name="treatment")
response_input = Concatenate(axis=1)([features, treatment])

est_response = architectures.feed_forward_net(response_input, Dense(1),
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
# Constant change in response so 
# just use one unit difference in
# treatment for counterfactual
####################################

DeepIV_tau_test = response_model.predict([X_test,np.ones_like(W_test)]) -\
                  response_model.predict([X_test,np.zeros_like(W_test)])

tau_test = data_generator.true_tau(X_test)

DeepIV_MSE = mean_squared_error(y_true = tau_test,
                                y_pred = DeepIV_tau_test.flatten())


params['DeepIV_MSE'] = DeepIV_MSE
dump(params, 'output/temp/params.pkl')
