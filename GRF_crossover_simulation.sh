#!/bin/bash

source /anaconda3/etc/profile.d/conda.sh 

# Fit GRF + IV series regression
conda activate rpy2
python -m simulations.GRF_simulation \
    -n $1 \
    -p $2 \
    -kappa $3 \
    -omega $4 \
    -additive $5 \
    -nuisance $6 \
    -s $7 \
    -CV $8
conda deactivate

# Fit DeepIV
conda activate DeepIV
python -m simulations.use_binary_DeepIV_on_GRF_simulation
