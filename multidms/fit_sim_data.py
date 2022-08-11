#!/usr/bin/env python


# python native
import pickle
from timeit import default_timer as timer
import os

# dependencies
import pandas as pd
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import sparse
import jaxopt

# local
from utils import *
from model import *

simulated_dataset = pd.read_csv("../results/simulated_dataset_v1.csv")
simulated_dataset_lib1 = simulated_dataset.query("library == 'lib_1'").copy()
simulated_dataset_lib1.aa_substitutions.fillna("", inplace=True)
simulated_dataset_lib1 = simulated_dataset_lib1.sample(n=15000, random_state=23)

homologs = json.load(open("../results/homolog_aa_seqs.json", "r"))
homologs["reference"] = homologs['1']
homologs["H2"] = homologs['2']
del homologs['1']
del homologs['2']

(X, y), updated_sim_func_score_df = create_homolog_modeling_data(
                            simulated_dataset_lib1, 
                            homologs, 
                            "homolog", 
                            "aa_substitutions",
                            "observed_phenotype"
                        )

# number of columns in the sparse arrays are the number of beta/shift params to fit
params = initialize_model_params(homologs, n_beta_shift_params = X["reference"].shape[1])

for key, value in params.items():
    if key == "α":
        for key_a, value_a in value.items():
            print(f"Parameter {key_a} has shape: {value_a.shape}")        
    else:
        print(f"Parameter {key} has shape: {value.shape}")

tol = 1e-6
maxiter = 20
solver = jaxopt.GradientDescent(cost_smooth, tol=tol, maxiter=maxiter)

start = timer()
params, state = solver.run(params, data=(X, y))
end = timer()

print(f"Full model optimization: {state.iter_num} iterations")
print(f"error = {state.error:.2e}")
print(f"cost = {cost_smooth(params, (X, y)):.2e}")
print(f"Wall time for fit: {end - start}")
