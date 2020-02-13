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

np.random.seed(1)

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
    result = ga.evolution_rosenbrock(
        settings_dict,
        param_dict,
        true_values,
        rt.prepare_run_params,
        rt.ensemble_fitness)
    print(':::::::::: Saving results :::::::::::::')
    universal.plot_costfunction(result['list_of_best_fitnesses'], output_dir_single)
    rt.plot_progress(result, true_values, output_dir_single)
    rt.plot_distance_history(result, true_values, output_dir_single)
    rt.plot_2d_location_progress(result, true_values, output_dir_single)
    rt.save_results(result, output_dir_single)


if __name__ == '__main__':
    main()
