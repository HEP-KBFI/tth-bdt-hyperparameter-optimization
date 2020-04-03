'''
Gradient descent optimization for the Rosenbrock function

Call with 'python'

Usage: rosenbrock_gradient_q.py
'''
import os
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import gradient_tools as gd
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
    settings_dict = universal.read_settings(settings_dir, 'gd')
    settings_dict.update(global_settings)
    param_file = os.path.join(settings_dir, 'rosenbrock_parameters.json')
    param_dict = universal.read_parameters(param_file)
    true_values = {'a': 1, 'b': 100}
    result_dict = gd.gradient_descent(
        settings_dict,
        param_dict,
        true_values,
        rt.initialize_values,
        rt.parameter_evaluation,
        rt.check_distance
    )
    print(':::::::::: Saving results :::::::::::::')
    gd.write_history(result_dict, output_dir)
    gd.contourplot(result_dict, true_values, param_dict, output_dir)
    gd.angle_plot(result_dict, output_dir)
    gd.step_plot(result_dict, output_dir)
    rt.plot_progress(result_dict, true_values, output_dir)
    rt.plot_distance_history(result_dict, true_values, output_dir)
    rt.plot_fitness_history(result_dict, output_dir)
    # rt.plot_2d_location_progress(result_dict, true_values, output_dir)
    rt.save_results(result_dict, output_dir)

if __name__ == '__main__':
    main()
