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
# Load the parameters 
########################

params = load('output/temp/params.pkl')

print(params)

########################
# Set model parameters
# loading CV results
########################

def get_dropout_l2(stage):
    
    selector = 'Stage 1 log loss' if (stage==1) else 'Stage 2 MSE'
    
    stage_results=pd.read_csv(f'output/MSEs/GRF_simulation_CV/STAGE{stage}_all_results.csv')

    stage_results = stage_results.loc[(
        (stage_results.n == params['n']) &\
        (stage_results.p == params['p']) &\
        (stage_results.kappa == params['kappa']) &\
        (stage_results.omega == params['omega']) &\
        (stage_results.additive == params['additive']) &\
        (stage_results.nuisance == params['nuisance'])
    )].sort_values(selector, ascending=True)

    dropout = stage_results.iloc[0].dropout
    l2 = stage_results.iloc[0].l2
    
    return dropout, l2

n = x.shape[0]
epochs = min(int(1500000./float(n)),300) # heuristic to select number of epochs
batch_size = 100
hidden = [128, 64, 32]
act = "relu"

if params['CV_regularizers']:
    stage1_dropout, stage1_l2 = get_dropout_l2(stage=1)
    stage2_dropout, stage2_l2 = get_dropout_l2(stage=2)
    print('Using tuned regularizer hyperparameters:')
    print(f'\tstage1_dropout, stage1_l2:{stage1_dropout}, {stage1_l2}')
    print(f'\tstage2_dropout, stage2_l2:{stage2_dropout}, {stage2_l2}')
else:
    l2 = 0.0001
    dropout_rate = min(1000./(1000. + n), 0.5)
    stage1_dropout, stage1_l2 = dropout_rate, l2
    stage2_dropout, stage2_l2 = dropout_rate, l2

# Build and fit treatment model
instruments = Input(shape=(z.shape[1],), name="instruments")
features = Input(shape=(x.shape[1],), name="features")
treatment_input = Concatenate(axis=1)([instruments, features])

est_treat = architectures.feed_forward_net(treatment_input,
                                           output=Dense(1, activation='sigmoid'),
                                           hidden_layers=hidden,
                                           dropout_rate=stage1_dropout, l2=stage1_l2,
                                           activations=act)

treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
treatment_model.compile('adam', loss="binary_crossentropy")
treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

# Build and fit response model

treatment = Input(shape=(t.shape[1],), name="treatment")
response_input = Concatenate(axis=1)([features, treatment])

est_response = architectures.feed_forward_net(response_input,
                                              output=Dense(1),
                                              activations=act,
                                              hidden_layers=hidden,
                                              dropout_rate=stage2_dropout,
                                              l2=stage2_l2)

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

params['DeepIV_MSE'] = DeepIV_MSE
params['stage1_dropout'] = stage1_dropout
params['stage1_l2'] = stage1_l2
params['stage2_dropout'] = stage2_dropout
params['stage2_l2'] = stage2_l2

pd.Series(params).to_frame().T.set_index('seed')\
  .to_csv(f"output/MSEs/GRF_simulation/CVregularizers_{params['CV_regularizers']}__" +\
          f'n_{params["n"]}__p_{params["p"]}__kappa_{params["kappa"]}__' +\
          f'omega_{params["omega"]}__additive_{params["additive"]}__nuisance_{params["nuisance"]}'+\
          f'seed_{params["seed"]}.csv')