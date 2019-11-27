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

#####################
# Generate data
#####################

n, ypcor, seed = 10000,0.5,75245
images = False

def datafunction(n, s, images=images, test=False):
    return data_generator.demand(n=n, seed=s, ypcor=ypcor,
                                 use_images=images, test=test)

X_train, Z_train, W_train, Y_train, g_train = datafunction(n, seed)
X_test, Z_test, W_test, Y_test, g_test = datafunction(n, seed+1)
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


#################################
# Plot distribution for sampled #
# test set instance             #
#################################

sample = np.random.choice(X_test.shape[0])

fig, ax = plt.subplots(figsize=(12,8))
ax.hist(treatment_model.sample([Z_test[[sample],:],X_test[[sample],:]],10000),bins=50,
        density=True,
        label='First stage DNN distribution\n$\\hat{p}_{\\theta}(T|X=x_i,Z=z_i)$')
ax.axvline(W_test[sample],color='red',label='Observed $T=t$')
ax.legend(loc='best')
ax.set_title('Visualizing the first stage distribution\n(For a single sampled $(x_i, z_i)$ test set instance)',size=22)
ax.set_xlabel('$T$')
ax.set_ylabel('Density')

plt.savefig('output/figures/first_stage_distribution.png')


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
    
    
sample = np.random.choice(X_test.shape[0],50)

fig, ax = plt.subplots(figsize=(12,6))

for s in sample:
    ax.plot(t_grid,gridded_counterfactual_estimates[s,:], alpha=0.5, color='grey', linewidth=0.5)

ax.set_ylabel('$\\hat{h}_\phi(x_i,t)$')
ax.set_xlabel('$t$')
ax.set_title('Estimated counterfactual prediction functions $\\hat{h}_\phi(x_i,t)$\n(sampled for N=50 test set instances, e.g. 50 $x_i$ values)')
plt.savefig('output/figures/linearization_justification.png')