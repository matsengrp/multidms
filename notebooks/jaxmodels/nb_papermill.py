import os
import multiprocessing
from functools import partial
import itertools as it
import argparse
import json

from papermill import execute_notebook


def explode_params_dict(params_dict):
    """
    Given a dictionary of model parameters,
    of which any of the values can be a list of values,
    compute all combinations of model parameter sets
    and returns a list of dictionaries representing each
    of the parameter sets.
    """
    varNames = sorted(params_dict)
    return [
        dict(zip(varNames, prod))
        for prod in it.product(*(params_dict[varName] for varName in varNames))
    ]


def execute_my_notebook(new_params, nb, results_dir):
    # create subdirectory for the results for each parameter set
    param_string = "--".join([f"{k}={v}" for k, v in new_params.items()])
    outdir = f"{results_dir}/{param_string}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    new_params["output_dir"] = outdir
    nb_path = f"{outdir}/{nb}"
    execute_notebook(nb, nb_path, parameters=new_params)


def parallel_execution(loop_params_dict, nb, results_dir, nb_workers=-1):
    # Define the number of processes to use
    num_processes = (
        multiprocessing.cpu_count() if nb_workers == -1 else nb_workers
    )  # Use number of CPU cores

    # Create a multiprocessing pool with the desired number of processes
    with multiprocessing.get_context("spawn").Pool(processes=num_processes) as pool:
        # Iterate over each set of loop parameters
        for params in loop_params_dict:
            # Apply the execute_notebook function to each set of parameters in parallel
            pool.apply_async(
                partial(execute_my_notebook, nb=nb, results_dir=results_dir),
                args=(params,),
            )

        # Close the pool and wait for all processes to complete
        pool.close()
        pool.join()


# Example of how to use parallel_execution
if __name__ == "__main__":
    # define defaults
    results = "results"
    nb = "my_noteboook.ipynb"
    params = "params.json"

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb", type=str, default=nb)
    parser.add_argument("--params", type=str, default=params)
    parser.add_argument("--nproc", type=int, default=-1)
    parser.add_argument("--output", type=str, default=results)
    args = parser.parse_args()

    if not os.path.exists(args.nb):
        raise ValueError(f"Notebook {args.nb} does not exist")
    if not os.path.exists(args.params):
        raise ValueError(f"Params file {args.params} does not exist")
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load the parameters from the JSON file
    with open(args.params, "r") as f:
        loop_params_dict = explode_params_dict(json.load(f))

        # Call parallel_execution to execute the notebooks in parallel
        parallel_execution(
            loop_params_dict, args.nb, args.output, nb_workers=args.nproc
        )
