'''
Particle swarm optimization for the hyperparameters optimization of XGBoost.
(MNIST numbers). Version for slurm.
Call with 'python3'

Usage: slurm_pso.py --sample_dir=DIR --nthread=INT --param_file=PTH --outputDir=DIR --mainDir=DIR

Options:
    --sample_dir=DIR        Directory of the sample
    --nthread=INT           Number of threads to use
    --param_file=PTH        Path to the parameters file
    --outputDir=DIR         Directory for plots and parameters
    --mainDir=DIR       Directory of the submit template

'''
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization.global_functions import prepare_run_params
from tthAnalysis.bdtHyperparameterOptimization.global_functions import calculate_improvement_wSTDEV
from tthAnalysis.bdtHyperparameterOptimization.global_functions import prepare_params_calc
from tthAnalysis.bdtHyperparameterOptimization.global_functions import create_datasets
from tthAnalysis.bdtHyperparameterOptimization.global_functions import read_parameters
from tthAnalysis.bdtHyperparameterOptimization.global_functions import save_results
import docopt
import os
from tthAnalysis.bdtHyperparameterOptimization.pso_main import find_bestFitnesses
from tthAnalysis.bdtHyperparameterOptimization.pso_main import calculate_personal_bests
from tthAnalysis.bdtHyperparameterOptimization.pso_main import read_weights
from tthAnalysis.bdtHyperparameterOptimization.pso_main import prepare_newDay
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import run_iteration

np.random.seed(1)


def run_pso(
    sample_dir,
    param_file,
    nthread,
    sample_size,
    w_init,
    w_fin,
    c1,
    c2,
    iterations,
    data_dict,
    value_dicts,
    templateDir
):
    parameter_dicts = prepare_run_params(
        nthread, value_dicts, sample_size)
    w = w_init
    w_step = (w_fin - w_init)/iterations
    new_parameters = parameter_dicts
    personal_bests = {}
    # improvements = []
    # improvement = 1
    compactness_threshold = 0.1
    compactness = calculate_improvement_wSTDEV(parameter_dicts)
    i = 1
    print(':::::::: Initializing :::::::::')
    fitnesses, pred_trains, pred_tests = run_iteration(
        outputDir, parameter_dicts, sample_dir,
        nthread, templateDir, sample_size
    )
    index = np.argmax(fitnesses)
    result_dict = {
        'data_dict': data_dict,
        'best_parameters': parameter_dicts[index],
        'pred_train': pred_trains[index],
        'pred_test': pred_tests[index],
        'best_fitness': max(fitnesses),
        'avg_scores': [np.mean(fitnesses)]
    }
    personal_bests = parameter_dicts
    best_fitnesses = fitnesses
    different_parameters = prepare_params_calc(
        result_dict['best_parameters'])
    current_speeds = np.zeros((sample_size, len(different_parameters)))
    while i <= iterations and compactness_threshold < compactness:
        print('::::::: Iteration: ', i, ' ::::::::')
        print(' --- Compactness: ', compactness, ' ---')
        parameter_dicts = new_parameters
        fitnesses, pred_trains, pred_tests = run_iteration(
            outputDir, parameter_dicts, sample_dir,
            nthread, templateDir, sample_size
        )
        best_fitnesses = find_bestFitnesses(fitnesses, best_fitnesses)
        personal_bests = calculate_personal_bests(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests)
        new_parameters = prepare_newDay(
            personal_bests, parameter_dicts,
            result_dict['best_parameters'],
            current_speeds, w, nthread, value_dicts
        )
        index = np.argmax(fitnesses)
        if result_dict['best_fitness'] < max(fitnesses):
            result_dict['best_parameters'] = parameter_dicts[index]
            result_dict['pred_train'] = pred_trains[index]
            result_dict['pred_test'] = pred_tests[index]
            result_dict['best_fitness'] = max(fitnesses)
        avg_scores = np.mean(fitnesses)
        result_dict['avg_scores'].append(avg_scores)
        compactness = calculate_improvement_wSTDEV(parameter_dicts)
        # improvements, improvement = calculate_improvement_wAVG(
        #     result_dict['avg_scores'],
        #     improvements,
        #     threshold
        # )
        w += w_step
        i += 1
    return result_dict


def main(param_file, nthread, sample_dir, outputDir, mainDir):
    templateDir = os.path.join(mainDir, "Slurm")
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
    result_dict = run_pso(
        sample_dir, param_file, nthread, sample_size,
        w_init, w_fin, c1, c2, iterations,
        data_dict, value_dicts, templateDir)
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