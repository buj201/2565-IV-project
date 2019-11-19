#####################
# Standard imports 
#####################

import argparse
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
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


def main(n, seed, ypcor, images=False):
    
    #####################
    # Generate data
    #####################

    def datafunction(n, s, images=images, test=False):
        return data_generator.demand(n=n, seed=s, ypcor=ypcor,
                                     use_images=images, test=test)

    X_train, Z_train, W_train, Y_train, _ = datafunction(n, seed)
    X_test, Z_test, W_test, Y_test, _ = datafunction(n, seed)
    tau_test = data_generator.true_tau(X_test)

    # Convenience renaming to match original DeepIV repo
    x, y, t, z = X_train, Y_train, W_train, Z_train

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
    # Build NNs
    ##########################################

    # NN hyperparameters
    dropout_rate = min(1000./(1000. + n), 0.5)
    epochs = int(1500000./float(n)) # heuristic to select number of epochs
    batch_size = 100
    hidden = [128, 64, 32]
    act = "relu"
    n_components = 10
    l2 = 0.001

    params = {
        'n': n,
        'seed': seed,
        'ypcor': ypcor,
        'dropout_rate': dropout_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'images': images,
        'hidden': hidden,
        'activation': act,
        'n_components': n_components,
        'l2': l2
    }

    # Build and fit treatment model

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

    #######################################
    # Estimate constant change in response 
    # using regression over linspace 100
    #######################################

    N_GRID = 100

    t_grid = np.linspace(np.percentile(t, 2.5),
                         np.percentile(t, 97.5),
                         N_GRID)

    gridded_counterfactual_estimates = np.zeros(shape=(X_test.shape[0],N_GRID))

    for ix in range(N_GRID):
        t_val = t_grid[ix]
        gridded_counterfactual_estimates[:,ix] = response_model.predict([X_test,np.ones_like(W_test)*t_val]).flatten()

    def get_coef(y, x=t_grid):
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        return slope

    DeepIV_tau_test = np.apply_along_axis(get_coef, 1, gridded_counterfactual_estimates)

    tau_test = data_generator.true_tau(X_test)

    DeepIV_MSE = mean_squared_error(y_true = tau_test,
                                    y_pred = DeepIV_tau_test)


    params['DeepIV_MSE'] = DeepIV_MSE
    dump(params, 'output/temp/params.pkl')

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='One run of low-D DeepIV')
    parser.add_argument('-n','--n_samples', help='Number of training samples', default=1000, type=int)
    parser.add_argument('-s','--seed', help='Random seed', default=1, type=int)
    parser.add_argument('-ypcor','--ypcor', help='Parameter controlling corr(y, p)', default=0.5, type=float)
    args = parser.parse_args()
    
    print(vars(args))
    
    main(args.n_samples, args.seed, args.ypcor, images=False)