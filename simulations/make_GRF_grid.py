from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import subprocess

N_ITER = 40

# Note -- currently just running with main effects to limit runtimes...
params = {'n_samples':[1000,2000],
          'p':[10,20],
          'kappa':[2,4],
          'omega':[0,1],
          'additive':[True,False],
          'nuisance':[True]}

params = pd.DataFrame(list(ParameterGrid(params)))
params = pd.concat(N_ITER*[params],axis=0)

np.random.seed(9755813)
params['seed'] = np.random.randint(low=1,high=4618907,size=params.shape[0])

params.to_csv('output/grids/GRF_grid.csv',index=False)

for ix, row in params.iterrows():
    print(row)
    seed = int(row['seed'])
    n = int(row['n_samples'])
    omega = int(row['omega'])
    additive = row['additive']
    nuisance = row['nuisance']
    kappa = row['kappa']
    p = row['p']
    file_name = 'output/MSEs/GRF_simulation/'+\
                f'n_{n}__p_{p}__kappa_{kappa}__' +\
                f'omega_{omega}__additive_{additive}__nuisance_{nuisance}__'+\
                f'seed_{seed}.csv'
    print(file_name)
    subprocess.run(["make", file_name, '--file=GRF_makefile', 
                    f'n={n}', f'p={p}', f'kappa={kappa}',
                    f'omega={omega}', f'additive={additive}',
                    f'nuisance={nuisance}', f'seed={seed}'])
    