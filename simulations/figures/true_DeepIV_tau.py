import sys
sys.path.append('../DeepIV/experiments/')

import data_generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook',font_scale=1.3)

t_grid_size = 1000
max_n = 7

x = np.concatenate([
    np.tile(np.linspace(0,10,t_grid_size),max_n)[:,np.newaxis],
    np.tile(data_generator.one_hot(np.arange(max_n),n_values=max_n),t_grid_size).T
],axis=1)

tau = data_generator.true_tau(x)

x_and_tau = np.concatenate([
    np.tile(np.linspace(0,10,t_grid_size),max_n)[:,np.newaxis],
    np.tile(np.arange(max_n),t_grid_size)[:,np.newaxis],
    tau[:,np.newaxis]
],axis=1)

x_and_tau = pd.DataFrame(x_and_tau)
x_and_tau.columns = ['$t$','$s$','$\\tau(t,s)$']

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(data=x_and_tau, x='$t$', y='$\\tau(t,s)$',
             hue='$s$',ax=ax)
ax.set_title('True treatment effect $\\tau(t,s)$',size=24)
plt.savefig('output/figures/true_DeepIV_tau.png')