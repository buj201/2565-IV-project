#!/bin/bash

source /anaconda3/etc/profile.d/conda.sh 

# Fit DeepIV
conda activate DeepIV
python -m simulations.DeepIV_simulation_no_MINST
conda deactivate

# Fit GRF + IV series regression
conda activate rpy2
python -m simulations.use_GRF_on_DeepIV_simulation_no_MINST

