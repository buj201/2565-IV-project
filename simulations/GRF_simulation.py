##########################
# R imports: Import R
# objects using rpy2
##########################

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
R = robjects.r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
GRF=importr('grf')

##########################
# Python imports
##########################

from joblib import dump

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_generators.GRF_simulations import *
from py_utils.rpy2_conversions import *

##########################
# Params -- refactor to
# pass in through CLI
##########################

p = 20
n = 5000
omega = 1 # Confounding: 1 = confounded, 0 is unconfounded
kappa = 4
additive = False
nuisance = True
n_test = 5000
seed = 6031

##########################
# Get sample and split
##########################

X, Y, W, Z, tau = get_sample(
    p, n + n_test, omega, kappa, additive, nuisance, seed
)
X_train = X[:n]
Y_train = Y[:n]
W_train = W[:n]
Z_train = Z[:n]
tau_train = tau[:n]

X_test = X[n:]
Y_test = Y[n:]
W_test = W[n:]
Z_test = Z[n:]
tau_test = tau[n:]

all_data = {
    'X_train': X_train,
    'Y_train': Y_train[:,np.newaxis],
    'W_train': W_train[:,np.newaxis],
    'Z_train': Z_train[:,np.newaxis],
    'tau_train': tau_train[:,np.newaxis],
    
    
    'X_test': X_test,
    'Y_test': Y_test[:,np.newaxis],
    'W_test': W_test[:,np.newaxis],
    'Z_test': Z_test[:,np.newaxis],
    'tau_test': tau_test[:,np.newaxis],
}

dump(all_data, 'output/temp/data_for_DeepIV.pkl')

##########################
# Fit forest
##########################

forest = GRF.instrumental_forest(X_train,
                                 to_FloatVector(Y_train),
                                 to_FloatVector(W_train),
                                 to_FloatVector(Z_train))
r_predict = robjects.r['predict']
forest_tau_test = to_array(r_predict(forest,X_test))
forest_MSE = mean_squared_error(y_true = tau_test,
                                y_pred = forest_tau_test.flatten())

##########################
# Fit IV series regression
##########################

import os
print(os.getcwd())
r_source = robjects.r['source']
r_source("./R_utils/iv_series_regression.R")
r_iv_series = robjects.globalenv['iv.series']

iv_series_tau_test = r_iv_series(
    X_train,
    to_FloatVector(Y_train),
    to_FloatVector(W_train),
    to_FloatVector(Z_train),
    X_test
)
iv_series_tau_test = to_array(iv_series_tau_test)
iv_series_MSE = mean_squared_error(y_true = tau_test,
                                   y_pred = iv_series_tau_test.flatten())

#############################
# Write output
#############################

params = {
    'p': p,
    'n': n,
    'omega': omega,
    'kappa': kappa,
    'additive': additive,
    'nuisance': nuisance,
    'n_test': n_test,
    'seed': seed,
    'iv_series_MSE': iv_series_MSE,
    'forest_MSE': forest_MSE
}


dump(params, 'output/temp/params.pkl')
