from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import subprocess

N_ITER = 40

params = {'n_samples':[1000,5000,10000,20000],
          'ypcor':[0.9,0.75,0.5,0.25,0.1]}

params = pd.DataFrame(list(ParameterGrid(params)))
params = pd.concat(N_ITER*[params],axis=0)

np.random.seed(9755813)
params['seed'] = np.random.randint(low=1,high=4618907,size=params.shape[0])

params.to_csv('output/grids/DeepIV_no_MINST_grid.csv',index=False)

for ix, row in params.iterrows():
    print(row)
    seed = int(row['seed'])
    n = int(row['n_samples'])
    ypcor = row['ypcor']
    file_name = 'output/MSEs/DeepIV_no_MINST_simulation/'+\
                f'n_{n}__seed_{seed}__ypcor_{ypcor}.csv'
    print(file_name)
    subprocess.run(["make", file_name, '--file=DeepIV_no_MINST_makefile',
                    f'seed={seed}', f'n={n}', f'ypcor={ypcor}'])
    