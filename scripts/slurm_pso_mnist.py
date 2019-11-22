'''
Particle swarm optimization for the hyperparameters optimization of XGBoost.
(MNIST numbers). Version for slurm.
Call with 'python'

Usage: slurm_pso_mnist.py --sample_dir=DIR --nthread=INT --paramDir=DIR --outputDir=DIR --mainDir=DIR

Options:
    --sample_dir=DIR        Directory of the sample
    --nthread=INT           Number of threads to use
    --paramDir=DIR          Path to the parameters file
    --outputDir=DIR         Directory for plots and parameters
    --mainDir=DIR           Directory of the main scripts (../hyper/hyper)

'''
from __future__ import division
import numpy as np
import xgboost as xgb
import docopt
import os
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader  as mf
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt

np.random.seed(1)

num_class = 10

def main(param_file, nthread, sample_dir, outputDir, mainDir):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    print('::::::: Loading data ::::::::')
    data_dict = mf.create_datasets(sample_dir, nthread)
    print('::::::: Reading parameters :::::::')
    param_file = os.path.join(paramDir, 'xgb_parameters.json')
    value_dicts = universal.read_parameters(param_file)
    weight_dict = universal.read_weights(value_dicts, paramDir)
    w_init = np.array(weight_dict['w_init'])
    w_fin = np.array(weight_dict['w_fin'])
    iterations = weight_dict['iterations']
    sample_size = weight_dict['sample_size']
    c1 = weight_dict['c1']
    c2 = weight_dict['c2']
    number_parameters = 7 # Read from file
    parameter_dicts = xt.prepare_run_params(
        nthread, value_dicts, sample_size)
    result_dict = pm.run_pso(
        sample_dir, nthread, sample_size,
        w_init, w_fin, c1, c2, iterations,
        data_dict, value_dicts, sm.run_iteration,
        number_parameters, parameter_dicts, outputDir,
        mainDir, num_class
    )
    universal.save_results(result_dict, outputDir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        paramDir = arguments['--paramDir']
        nthread = int(arguments['--nthread'])
        sample_dir = arguments['--sample_dir']
        outputDir = arguments['--outputDir']
        mainDir = arguments['--mainDir']
        main(paramDir, nthread, sample_dir, outputDir, mainDir)
    except docopt.DocoptExit as e:
        print(e)