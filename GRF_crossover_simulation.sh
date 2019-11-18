#!/bin/bash

source /anaconda3/etc/profile.d/conda.sh 

# Fit GRF + IV series regression
conda activate rpy2
python -m simulations.GRF_simulation
conda deactivate

# Fit DeepIV
conda activate DeepIV
python -m simulations.use_DeepIV_on_GRF_simulation

