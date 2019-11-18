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

from joblib import load

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from py_utils.rpy2_conversions import *

##########################
# Load data
##########################

all_data = load('output/temp/data_for_GRF.pkl')
locals().update(all_data)

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
print('Forest null predictions:', np.isnan(forest_tau_test).sum())
print('Forest predictions size:', forest_tau_test.shape)

##########################
# Fit IV series regression
##########################

import os
print(os.getcwd())
r_source = robjects.r['source']
r_source("./R_utils/iv_series_regression.R")
r_iv_series = robjects.globalenv['iv.series']

print(X_train.shape)
print(Y_train.shape)
print(W_train.shape)
print(Z_train.shape)
print(X_test.shape)

iv_series_tau_test = r_iv_series(
    X_train,
    to_FloatVector(Y_train),
    to_FloatVector(W_train),
    to_FloatVector(Z_train),
    X_test
)
iv_series_tau_test = to_array(iv_series_tau_test)
iv_series_MSE = mean_squared_error(y_true = tau_test,
                                   y_pred = iv_series_tau_test)

#############################
# Write output
#############################

params = load('output/temp/params.pkl')

params['forest_MSE'] = forest_MSE
params['iv_series_MSE'] = iv_series_MSE

pd.Series(params).to_frame().T.set_index('seed')\
  .to_csv('output/MSEs/MSE_test_2.csv')