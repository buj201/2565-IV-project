##########################################################################
# Adapted from jhartford/DeepIV/experiments/demand_simulation.py
##########################################################################

from sklearn.metrics import mean_squared_error, log_loss

import warnings

import sys
sys.path.append('../')

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

from .BinaryTreatmentModel import make_shared_net, add0, add1

import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.merge import Concatenate

from keras.layers import Input, Dense, Add, multiply, Activation
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate

from keras import regularizers

from keras.backend import zeros, ones
import keras.backend as K
from keras.layers.core import Lambda

import numpy as np
import pandas as pd


def stage1_optimize(dropout_rates, l2s):
    results = []
    for dropout_rate in dropout_rates:
        for l2 in l2s:
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
            treatment_model.compile('adam', loss="binary_crossentropy")

            treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)


            # Get validation log loss
            test_treatment_preds = treatment_model.predict([Z_test, X_test])
            DeepIV_log_loss = log_loss(y_true=W_test, y_pred = test_treatment_preds)
            results.append({'dropout':dropout_rate, 'l2':l2, 'Stage 1 log loss': DeepIV_log_loss})
            
    return pd.DataFrame(results)



def stage2_optimize(best_stage1_dropout, best_stage1_l2, dropout_rates, l2s):
    
    # Build and fit treatment model
    instruments = Input(shape=(z.shape[1],), name="instruments")
    features = Input(shape=(x.shape[1],), name="features")
    treatment_input = Concatenate(axis=1)([instruments, features])

    est_treat = architectures.feed_forward_net(treatment_input,
                                               output=Dense(1, activation='sigmoid'),
                                               hidden_layers=hidden,
                                               dropout_rate=best_stage1_dropout, l2=best_stage1_l2,
                                               activations=act)
    
    treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
    treatment_model.compile('adam', loss="binary_crossentropy")
    treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

    pi1 = treatment_model.predict([z, x])
    pi0 = (1-pi1)
    
    pi1_test = treatment_model.predict([Z_test, X_test])
    pi0_test = (1-pi1_test)
        
    results = []
    for dropout_rate in dropout_rates:
        for l2 in l2s:
            # Optimize response model
            features = Input(shape=(x.shape[1],), name="features")
            pi0_input = Input(shape=(pi0.shape[1],), name="pi0")
            pi1_input = Input(shape=(pi1.shape[1],), name="pi1")

            response_input0 = Lambda(add0, name='Add0')(features)
            response_input1 = Lambda(add1, name='Add1')(features)

            shared = make_shared_net(x.shape[1]+1, act, hidden, dropout_rate=dropout_rate, l2=l2)

            h0 = shared(response_input0)
            h1 = shared(response_input1)

            weighted_yhat = Add(name='weighted_yhat')([
                multiply([pi0_input, h0]),
                multiply([pi1_input, h1]),
            ])

            response_model = Model(inputs=[features, pi0_input, pi1_input],
                                   outputs=weighted_yhat)
            response_model.compile('adam', loss='mse')

            # Now fit the model

            response_model.fit([x, pi0, pi1], y, epochs=epochs, verbose=1,
                               batch_size=batch_size)

            ####################################
            # The GRF simulation has a binary
            # treatment W -- so the treatment effect
            # is the difference h(x,w=1) - h(x,w=0)
            ####################################

            add_out_func = K.function(inputs = response_model.input,
                                      outputs = [response_model.get_layer('weighted_yhat').output])
            
            hhat1 = add_out_func([X_test, np.zeros_like(pi0_test), np.ones_like(pi1_test)])[0]
            hhat0 = add_out_func([X_test, np.ones_like(pi0_test), np.zeros_like(pi1_test)])[0]

            DeepIV_tau_test = (hhat1 - hhat0).flatten()
            DeepIV_causal_MSE = mean_squared_error(y_true = tau_test,
                                                   y_pred = DeepIV_tau_test)
            
            DeepIV_stage2_MSE = mean_squared_error(y_true = Y_test,
                                                   y_pred = add_out_func([X_test, pi0_test, pi1_test])[0])

            
            results.append({'dropout':dropout_rate, 'l2':l2, 
                            'Causal MSE':DeepIV_causal_MSE, 
                            'Stage 2 MSE': DeepIV_stage2_MSE})

    return pd.DataFrame(results)


if __name__ == '__main__':

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

    params = load('output/temp/params.pkl')
    
    ########################
    # Set model parameters
    ########################

    n = x.shape[0]
    epochs = min(int(1500000./float(n)),300) # heuristic to select number of epochs
    batch_size = 100
    hidden = [128, 64, 32]
    act = "relu"

    dropout_rates = [min(1000./(1000. + n), 0.5)] # Use default 
    l2s = np.logspace(-3,np.log10(0.25),8,10)
    
    ######################
    # Run stage 1 search #    
    ######################
    
    stage1_results=stage1_optimize(dropout_rates, l2s)    
    best_stage1_params = stage1_results.loc[stage1_results['Stage 1 log loss'].idxmin()]
    print(best_stage1_params)
    best_stage1_dropout = best_stage1_params['dropout']
    best_stage1_l2 = best_stage1_params['l2']
    
    pd.DataFrame(stage1_results)\
      .to_csv('output/MSEs/GRF_simulation_CV/CVregularizers_True__'+\
              'STAGE1__'+\
              f'n_{params["n"]}__p_{params["p"]}__kappa_{params["kappa"]}__' +\
              f'omega_{params["omega"]}__additive_{params["additive"]}__nuisance_{params["nuisance"]}.csv',
              index=False)
    
    ######################
    # Run stage 2 search #    
    ######################
    
    stage2_results = stage2_optimize(best_stage1_dropout, best_stage1_l2, dropout_rates, l2s)
    
    pd.DataFrame(stage2_results)\
      .to_csv('output/MSEs/GRF_simulation_CV/CVregularizers_True__'+\
              'STAGE2__'+\
              f'n_{params["n"]}__p_{params["p"]}__kappa_{params["kappa"]}__' +\
              f'omega_{params["omega"]}__additive_{params["additive"]}__nuisance_{params["nuisance"]}.csv',
              index=False)