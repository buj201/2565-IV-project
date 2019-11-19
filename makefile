SHELL = /bin/bash

no_MINST_prereqs = simulations/DeepIV_simulation_no_MINST.py \
				   simulations/use_GRF_on_DeepIV_simulation_no_MINST.py \
				   R_utils/iv_series_regression.R \
				   DeepIV_crossover_simulation.sh
				   

output/MSEs/DeepIV_no_MINST_simulation/n_$(n)__seed_$(seed)__ypcor_$(ypcor).csv: $(no_MINST_prereqs)
	./DeepIV_crossover_simulation.sh $(n) $(seed) $(ypcor)
