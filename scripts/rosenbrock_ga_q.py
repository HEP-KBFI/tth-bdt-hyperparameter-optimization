'''
Testing the genetic algorithm via finding the global minimum of the Rosenbrock function.
Call with 'python'

Usage: rosenbrock_ga_q.py
'''
import os
import json
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_main as ga
from tthAnalysis.bdtHyperparameterOptimization import rosenbrock_tools as rt

NUMBER_REPETITIONS = 10

def main():
    print('::::::: Reading settings and parameters :::::::')
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
    settings_dir = os.path.join(
        main_dir, 'data')
    global_settings = universal.read_settings(settings_dir, 'global')
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    settings_dict = universal.read_settings(settings_dir, 'ga')
    settings_dict.update(global_settings)
    param_file = os.path.join(
        settings_dir, 'rosenbrock_parameters.json')
    param_dict = universal.read_parameters(param_file)
    true_values = {'a': 1, 'b': 100}

    result_dicts = []
    for i in range(NUMBER_REPETITIONS):
        output_dir_single = os.path.join(output_dir, 'iteration_' + str(i))
        if not os.path.isdir(output_dir_single):
            os.makedirs(output_dir_single)
        np.random.seed(i)
        result = ga.evolution_rosenbrock(
            settings_dict,
            param_dict,
            true_values,
            rt.prepare_run_params,
            rt.ensemble_fitness)
        universal.plot_costfunction(result['list_of_best_fitnesses'], output_dir_single)
        rt.plot_progress(result, true_values, output_dir_single)
        rt.plot_distance_history(result, true_values, output_dir_single)
        rt.plot_2d_location_progress(result, true_values, output_dir_single)
        rt.save_results(result, output_dir_single)
        print(
            'Results of iteration '
            + str(i) + ' are saved to ' + str(output_dir)
        )
        result_dicts.append(result)
    print(':::::::::: Saving results :::::::::::::')
    best_fitnesses = [
        result_dict['best_fitness'] for result_dict in result_dicts
    ]
    fitness_scores_path = os.path.join(output_dir, 'fitness_scores.json')
    with open(fitness_scores_path, 'w') as file:
        json.dump(best_fitnesses, file)


if __name__ == '__main__':
    main()
