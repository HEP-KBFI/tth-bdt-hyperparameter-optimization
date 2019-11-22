'''
Particle swarm optimization for the hyperparameters optimization of XGBoost.
(MNIST numbers). Version for quasar.
Call with 'python3'

Usage: quasar_pso_mnist.py --sample_dir=DIR --nthread=INT --param_file=PTH --outputDir=DIR --mainDir=DIR

Options:
    --sample_dir=DIR        Directory of the sample
    --nthread=INT           Number of threads to use
    --param_file=PTH        Path to the parameters file
    --outputDir=DIR         Directory for plots and parameters
    --mainDir=DIR           Directory of the main scripts (../hyper/hyper)

'''
from __future__ import division
import numpy as np
import xgboost as xgb
import docopt
import os
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader  as mf
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
np.random.seed(1)


def main(param_file, nthread, sample_dir, outputDir, mainDir):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    print("::::::: Loading data ::::::::")
    data_dict = create_datasets(sample_dir, nthread)
    print("::::::: Reading parameters :::::::")
    value_dicts = universal.read_parameters(param_file)
    pso_settings = pm.read_weights(value_dicts)
    global_settings = universal.read_settings('global')
    iterations = pso_settings['iterations']
    sample_size = pso_settings['sample_size']
    parameter_dicts = xt.prepare_run_params(
        nthread, value_dicts, sample_size)
    result_dict = pm.run_pso(
        global_settings, pso_settings, data_dict,
        value_dicts, xt.ensemble_fitnesses, parameter_dicts
    )
    universal.save_results(result_dict, outputDir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        param_file = arguments['--param_file']
        nthread = int(arguments['--nthread'])
        sample_dir = arguments['--sample_dir']
        outputDir = arguments['--outputDir']
        mainDir = arguments['--mainDir']
        main(param_file, nthread, sample_dir, outputDir, mainDir)
    except docopt.DocoptExit as e:
        print(e)

# import what scoring to use here ??