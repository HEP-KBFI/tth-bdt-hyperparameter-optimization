'''
Particle swarm optimization for the hyperparameters optimization of XGBoost.
(MNIST numbers). Version for slurm.
Call with 'python'

Usage: slurm_pso_mnist.py
'''
import numpy as np
import os
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import rosenbrock_tools as rt

np.random.seed(1)


def main():
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
    universal.save_run_settings(output_dir)
    pso_settings = pm.read_weights(settings_dir)
    print("::::::: Reading parameters :::::::")
    param_file = os.path.join(settings_dir, 'rosenbrock_parameters.json')
    value_dicts = universal.read_parameters(param_file)
    parameter_dicts = rt.prepare_run_params(
        value_dicts, pso_settings['sample_size'])
    true_values = {'a': 1, 'b': 100}
    result_dict = rt.run_pso(
        parameter_dicts,
        true_values,
        value_dicts,
        output_dir,
        global_settings
    )
    print(':::::::::: Saving results :::::::::::::')
    rt.plot_progress(result_dict, true_values, output_dir)
    rt.plot_distance_history(result_dict, true_values, output_dir)
    rt.plot_fitness_history(result_dict, output_dir)
    rt.plot_2d_location_progress(result_dict, true_values, output_dir)
    rt.save_results(result_dict, output_dir)

if __name__ == '__main__':
    main()