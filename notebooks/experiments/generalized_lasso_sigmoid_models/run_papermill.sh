#!/usr/bin/env bash

/usr/bin/time python nb_papermill.py \
    --nb fit_generalized_lasso.ipynb \
    --params params.json \
    --output test_run_results
