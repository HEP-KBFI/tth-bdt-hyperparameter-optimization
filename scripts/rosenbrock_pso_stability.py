'''
PSO stability check based on Rosenbrock function.

Call with 'python'
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
    for i in range(1000):
        np.random.seed(i)
        global_settings['output_dir'] = os.path.join(output_dir, str(i))
        if not os.path.exists(global_settings['output_dir']):
            os.makedirs(global_settings['output_dir'])
        print(global_settings['output_dir'])
        result = rt.run_pso(
            parameter_dicts,
            true_values,
            value_dicts,
            output_dir,
            global_settings,
            plot_pso_location=False
        )
        print(':::::::::: Saving results of iteration %s :::::::::::::') %i
        best_parameters_list.append(result['best_parameters'])
        rt.save_results(result, global_settings['output_dir'])
    best_fitnesses_list = create_fitness_list(best_parameters_list)
    produce_stability_plots(
        best_parameters_list, best_fitnesses_list, output_dir)
    return best_parameters_list, best_parameters_list


def create_fitness_list(best_parameters_list):
    best_fitnesses_list = []
    for best_parameters in best_parameters_list:
        best_fitnesses_list.append(rt.parameter_evaluation(best_parameters))
    return best_fitnesses_list


def produce_stability_plots(
        best_parameters_list,
        best_fitnesses_list,
        output_dir
):
    x_distances = np.array([np.abs(i['x'] - 1) for i in best_parameters_list])
    y_distances = np.array([np.abs(i['y'] - 1) for i in best_parameters_list])
    absolute_distances = np.sqrt(x_distances**2 + y_distances**2)
    plot_absolute_distances(absolute_distances, output_dir)
    plot_fitness_values(best_fitnesses_list, output_dir)


def plot_absolute_distances(absolute_distances, output_dir):
    plt.hist(absolute_distances, histtype='step')
    plt.title("Absolute distance from minimum")
    plt.xlabel("Distance")
    plt.ylabel("# cases")
    output_path = os.path.join(output_dir, 'absolute_distances.png')
    plt.savefig(output_path, bbox_inches='tight')


def plot_fitness_values(best_fitnesses_list, output_dir):
    plt.hist(best_fitnesses_list, histtype='step')
    plt.title("Fitness values")
    plt.xlabel("Found minimum value")
    plt.ylabel("# cases")
    output_path = os.path.join(output_dir, 'fitness_values.png')
    plt.savefig(output_path, bbox_inches='tight')


if __name__ == '__main__':
    best_parameters_list, best_fitnesses_list = main()
