'''
Testing the genetic algorithm via finding the global minimum of the Rosenbrock function.
Call with 'python'

Usage: rosenbrock_ga_q.py
'''
import os
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_main as ga
from tthAnalysis.bdtHyperparameterOptimization import rosenbrock_tools as rt


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
    output_dir = settings_dict['output_dir']
    true_values = {'a': 1, 'b': 100}
    best_parameters_list = []
    for i in range(100):
        np.random.seed(i)
        settings_dict['output_dir'] = os.path.join(output_dir, str(i))
        print(settings_dict['output_dir'])
        result = ga.evolution_rosenbrock(
            settings_dict,
            param_dict,
            true_values,
            rt.prepare_run_params,
            rt.ensemble_fitness)
        print(':::::::::: Saving results of iteration %s :::::::::::::') %i
        best_parameters_list.append(result['best_parameters'])
        rt.save_results(result, settings_dict['output_dir'])
    best_fitnesses_list = create_fitness_list(best_parameters_list)
    return best_parameters_list, best_parameters_list


def create_fitness_list(best_parameters_list):
    best_fitnesses_list = []
    for best_parameters in best_parameters_list:
        best_fitnesses_list.append(rt.parameter_evaluation(best_parameters))
    return best_fitnesses_list


if __name__ == '__main__':
    best_parameters_list, best_parameters_list = main()
