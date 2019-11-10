'''
Grid search for best parameters to be compared with evol. algorithms
Call with 'python3'

Usage: xgb_gridSearch.py --sample_dir=DIR --nthread=INT --param_file=PTH --outputDir=DIR

Options:
    --sample_dir=DIR        Directory of the sample
    --nthread=INT           Number of threads to use
    --param_file=PTH        Path to the parameters file
    --outputDir=DIR         Directory for plots and parameters

'''
from tthAnalysis.bdtHyperparameterOptimization.universal import read_parameters
from tthAnalysis.bdtHyperparameterOptimization.universal import save_results
from tthAnalysis.bdtHyperparameterOptimization.mnist_filereader import create_datasets
from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import ensemble_fitnesses
import docopt
from itertools import product
import numpy as np
import os

GRID_SIZE = 2

def create_all_combinations(nr_parameters, grid_size):
    combinations = list(product(
        range(grid_size),
        repeat=nr_parameters)
    )
    return combinations


def initialize_values(parameters, grid_size):
    parameter_dicts = []
    combinations = create_all_combinations(len(parameters), grid_size)
    for iterations in combinations:
        parameter_dict = single_paramSet(parameters, iterations, grid_size)
        parameter_dicts.append(parameter_dict)
    return parameter_dicts


def single_paramSet(parameters, iterations, grid_size):
    parameter_dict = {}
    for param, iteration in zip(parameters, iterations):
        key = param['p_name']
        range_size = param['range_end'] - param['range_start']
        if grid_size == 1:
            value = (param['range_end'] + param['range_start']) / 2
        else:
            step_size = range_size / (grid_size - 1)
            value = param['range_start'] + (iteration * step_size)
        parameter_dict[key] = value
        if param['true_int'] == 'True':
            parameter_dict[key] = int(np.ceil(value))
    return parameter_dict


def param_update(parameter_dicts, nthread):
    run_params = []
    params = {
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': nthread,
        'seed': 1,
    }
    for param_dict in parameter_dicts:
        param_dict.update(params)
        run_params.append(param_dict)
    return run_params


def main(param_file, nthread, sample_dir, outputDir):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    parameters = read_parameters(param_file)
    parameter_dicts = initialize_values(parameters, GRID_SIZE)
    parameter_dicts = param_update(parameter_dicts, nthread)
    data_dict = create_datasets(sample_dir, nthread)
    fitnesses, pred_trains, pred_tests = ensemble_fitnesses(
        parameter_dicts, data_dict
    )
    index =  np.argmax(fitnesses)
    result_dict = {
        'best_parameters': parameter_dicts[index],
        'pred_train': pred_trains[index],
        'pred_test': pred_tests[index],
        'data_dict': data_dict
    }
    save_results(result_dict, outputDir, roc=False)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        param_file = arguments['--param_file']
        nthread = int(arguments['--nthread'])
        sample_dir = arguments['--sample_dir']
        outputDir = arguments['--outputDir']
    except docopt.DocoptExit as e:
        print(e)
    main(param_file, nthread, sample_dir, outputDir)