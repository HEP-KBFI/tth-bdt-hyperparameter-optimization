'''
Particle swarm optimization for the hyperparameters optimization of XGBoost.
(MNIST numbers). Version for quasar.
Call with 'python3'

Usage: quasar_pso.py --sample_dir=DIR --nthread=INT --param_file=PTH --outputDir=DIR --mainDir=DIR

Options:
    --sample_dir=DIR        Directory of the sample
    --nthread=INT           Number of threads to use
    --param_file=PTH        Path to the parameters file
    --outputDir=DIR         Directory for plots and parameters
    --mainDir=DIR           Directory of the main scripts (../hyper/hyper)

'''
import numpy as np
import xgboost as xgb
import docopt
import os
from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import ensemble_fitnesses
from tthAnalysis.bdtHyperparameterOptimization.mnist_filereader import create_datasets
from tthAnalysis.bdtHyperparameterOptimization.universal import read_parameters
from tthAnalysis.bdtHyperparameterOptimization.universal import save_results
from tthAnalysis.bdtHyperparameterOptimization.pso_main import read_weights
from tthAnalysis.bdtHyperparameterOptimization.pso_main import run_pso
from tthAnalysis.bdtHyperparameterOptimization.pso_main import prepare_run_params

np.random.seed(1)


def main(param_file, nthread, sample_dir, outputDir, mainDir):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    print("::::::: Loading data ::::::::")
    data_dict = create_datasets(sample_dir, nthread)
    print("::::::: Reading parameters :::::::")
    value_dicts = read_parameters(param_file)
    weight_dict = read_weights(value_dicts, mainDir)
    w_init = np.array(weight_dict['w_init'])
    w_fin = np.array(weight_dict['w_fin'])
    iterations = weight_dict['iterations']
    sample_size = weight_dict['sample_size']
    c1 = weight_dict['c1']
    c2 = weight_dict['c2']
    number_parameters = 7 # Read from file
    parameter_dicts = prepare_run_params(
        nthread, value_dicts, sample_size)
    result_dict = run_pso(
        sample_dir, nthread, sample_size,
        w_init, w_fin, c1, c2, iterations,
        data_dict, value_dicts, ensemble_fitnesses,
        number_parameters, parameter_dicts
    )
    save_results(result_dict, outputDir)


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