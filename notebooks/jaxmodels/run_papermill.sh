#!/usr/bin/env bash
#===================================
# usage: nb_papermill.py [-h] [--nb NB] [--params PARAMS] [--nproc NPROC] [--output OUTPUT]
# 
# options:
#   -h, --help       show this help message and exit
#   --nb NB
#   --params PARAMS
#   --nproc NPROC
#   --output OUTPUT
#===================================

/usr/bin/time python nb_papermill.py \
    --nproc 2 \
    --nb jaxmodels.ipynb \
    --params params.json \
    --output papermill_results
