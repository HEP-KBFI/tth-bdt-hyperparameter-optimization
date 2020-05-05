'''Run stability and/or performance for different methods of finding a minima
of the Rosenbrock function.

Call with 'python'

Usage: slurm_fitness.py  --output_dir=DIR --choice=STR --method=STR

Options:

    --output_dir=DIR         Directory of the output
    --choice=STR             Either to run 'stability', 'performance' or 'both'
    --method=STR             Either to run 'ga', 'pso', 'gd', 'all'
'''

import numpy as np
import os
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import rosenbrock_tools as rt
from tthAnalysis.bdtHyperparameterOptimization import gradient_tools as gd
from tthAnalysis.bdtHyperparameterOptimization import ga_main as ga
import matplotlib.pyplot as plt
import docopt
import matplotlib.ticker as ticker


def main(choice, method, output_dir):
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
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    param_file = os.path.join(settings_dir, 'rosenbrock_parameters.json')
    value_dicts = universal.read_parameters(param_file)
    gd_settings = universal.read_settings(settings_dir, 'gd')
    pso_settings = universal.read_settings(settings_dir, 'pso')
    ga_settings = universal.read_settings(settings_dir, 'ga')
    gd_settings.update(global_settings)
    ga_settings.update(global_settings)
    pso_settings.update(global_settings)
    parameter_dicts = rt.prepare_run_params(
        value_dicts, pso_settings['sample_size'])
    true_values = {'a': 1, 'b': 100}
    if choice == 'performance' or choice == 'both':
        print("Testing performance")
        result_dict = run_single_choice(
                ga_settings,
                gd_settings,
                pso_settings,
                parameter_dicts,
                true_values,
                value_dicts,
                method
        )
        plot_performance_main(result_dict, 'distance', output_dir, value_dicts)
        plot_performance_main(result_dict, 'fitness', output_dir, value_dicts)
    if choice == 'stability' or choice == 'both':
        i = 0
        result_dicts = []
        print("Testing stability")
        while i < 1000:
            np.random.seed(i)
            result_dict = run_single_choice(
                ga_settings,
                gd_settings,
                pso_settings,
                parameter_dicts,
                true_values,
                value_dicts,
                method
            )
            result_dicts.append(result_dict)
            i += 1
        plot_stabilities_main(result_dicts, method, 'distance', output_dir)
        plot_stabilities_main(result_dicts, method, 'fitness', output_dir)


def run_single_choice(
        ga_settings,
        gd_settings,
        pso_settings,
        parameter_dicts,
        true_values,
        value_dicts,
        method
):
    result_dict = {}
    print("Random choice")
    result_dict['random_result'] = rt.run_random(
        parameter_dicts,
        true_values,
        value_dicts,
        pso_settings
    )
    if method == 'ga' or method == 'all':
        print("Genetic Algorithm")
        result_dict['ga_result'] = ga.evolution_rosenbrock(
            ga_settings,
            value_dicts,
            true_values,
            rt.prepare_run_params,
            rt.ensemble_fitness)
    if method == 'pso' or method == 'all':
        print("Particle swarm optimization")
        result_dict['pso_result'] = rt.run_pso(
            parameter_dicts,
            true_values,
            value_dicts,
            pso_settings,
        )
    if method == 'gd' or method == 'all':
        print("Gradient descent")
        result_dict['gd_result'] = gd.gradient_descent(
            gd_settings,
            value_dicts,
            true_values,
            rt.initialize_values,
            rt.parameter_evaluation,
            rt.check_distance
        )
    return result_dict

####################################


def plot_stabilities_main(
        result_dicts, method, to_plot, output_dir
):
    random_dicts = [d['random_result'] for d in result_dicts]
    rnd_best_param_list = [d['best_parameters'] for d in random_dicts]
    rnd_best_fitnesses = [d['best_fitness'] for d in random_dicts]
    produce_stability_plots(
            rnd_best_param_list,
            rnd_best_fitnesses,
            output_dir,
            to_plot,
            'RND',
            rnd=True
    )
    if method == 'ga' or method == 'all':
        ga_dicts = [d['ga_result'] for d in result_dicts]
        ga_best_param_list = [d['best_parameters'] for d in ga_dicts]
        ga_best_fitnesses = [d['best_fitness'] for d in ga_dicts]
        produce_stability_plots(
                ga_best_param_list,
                ga_best_fitnesses,
                output_dir,
                to_plot,
                'GA'
        )
    elif method == 'pso' or method == 'all':
        pso_dicts = [d['pso_result'] for d in result_dicts]
        pso_best_param_list = [d['best_parameters'] for d in pso_dicts]
        pso_best_fitnesses = [d['best_fitness'] for d in pso_dicts]
        produce_stability_plots(
                pso_best_param_list,
                pso_best_fitnesses,
                output_dir,
                to_plot,
                'PSO'
        )
    elif method == 'gd' or method == 'all':
        gd_dicts = [d['gd_result'] for d in result_dicts]
        gd_best_param_list = [d['best_parameters'] for d in gd_dicts]
        gd_best_fitnesses = [d['best_fitness'] for d in gd_dicts]
        produce_stability_plots(
                gd_best_param_list,
                gd_best_fitnesses,
                output_dir,
                to_plot,
                'Gradient descent'
        )
    output_path = os.path.join(output_dir, 'best_' + to_plot + '_stability.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def create_fitness_list(best_parameters_list):
    best_fitnesses_list = []
    for best_parameters in best_parameters_list:
        best_fitnesses_list.append(rt.parameter_evaluation(best_parameters))
    return best_fitnesses_list


def produce_stability_plots(
        best_parameters_list,
        best_fitnesses_list,
        output_dir,
        to_plot,
        label,
        rnd=False
):
    x_distances = np.array([np.abs(i['x'] - 1) for i in best_parameters_list])
    y_distances = np.array([np.abs(i['y'] - 1) for i in best_parameters_list])
    absolute_distances = np.sqrt(x_distances**2 + y_distances**2)
    if to_plot == 'distance':
        plot_absolute_distances(absolute_distances, rnd, label)
    if to_plot == 'fitness':
        plot_fitness_values(best_fitnesses_list, rnd, label)


def plot_absolute_distances(absolute_distances, rnd, label):
    plt.hist(
        absolute_distances,
        histtype='step',
        bins=int(np.ceil(np.sqrt(len(absolute_distances)))),
        label=label
    )
    if rnd:
        plt.title("Absolute distance from minimum")
        plt.xlabel("Distance")
        plt.ylabel("# cases")
        plt.legend()
        plt.yscale('log')


def plot_fitness_values(best_fitnesses_list, rnd, label):
    plt.hist(
        best_fitnesses_list,
        histtype='step',
        bins=int(np.ceil(np.sqrt(len(best_fitnesses_list)))),
        label=label
    )
    if rnd:
        plt.title("Fitness values")
        plt.xlabel("Found minimum value")
        plt.ylabel("# cases")
        plt.legend()
        plt.yscale('log')


#####################################################################

def plot_performance_main(result_dict, to_plot, output_dir, value_dicts):
    random_result = result_dict['random_result']
    plotting_main(random_result, to_plot, 'Rnd', rnd=True)
    if method == 'ga' or method == 'all':
        ga_result = result_dict['ga_result']
        plotting_main(ga_result, to_plot, 'GA')
    elif method == 'pso' or method == 'all':
        pso_result = result_dict['pso_result']
        plotting_main(pso_result, to_plot, 'PSO')
    output_path = os.path.join(
        output_dir, 'best_' + to_plot + '_performance.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
    if method == 'gd' or method == 'all':
        gd_result = result_dict['gd_result']
        plot_gd(gd_result, output_dir, value_dicts)


def plotting_main(result_dict, to_plot, label, rnd=False):
    if to_plot == 'distance':
        plot_distances(result_dict, rnd)
    elif to_plot == 'fitness':
        plot_fitnesses_history(result_dict, rnd)


def plot_fitnesses_history(result_dict, rnd):
    old_fitnesses = result_dict['list_of_best_fitnesses']
    x_values = np.arange(len(old_fitnesses))
    plt.plot(
        x_values,
        old_fitnesses,
        label='Best fitness'
    )
    if rnd:
        plt.xlabel('Iteration number / #')
        plt.ylabel('Fitness')
        axis = plt.gca()
        axis.set_aspect('auto', adjustable='box')
        axis.xaxis.set_major_locator(ticker.AutoLocator())
        plt.grid(True)
        plt.legend()
        plt.yscale('log')
        plt.tick_params(top=True, right=True, direction='in')


def plot_distances(result_dict, rnd):
    best_parameters_list = result_dict['list_of_old_bests']
    x_distances = np.array([np.abs(i['x'] - 1) for i in best_parameters_list])
    y_distances = np.array([np.abs(i['y'] - 1) for i in best_parameters_list])
    absolute_distances = np.sqrt(x_distances**2 + y_distances**2)
    x_values = np.arange(len(absolute_distances))
    plt.plot(
        x_values,
        absolute_distances,
        label='Distance to minimum')
    if rnd:
        plt.xlabel('Iteration number / #')
        plt.ylabel('Distance')
        axis = plt.gca()
        axis.set_aspect('auto', adjustable='box')
        axis.xaxis.set_major_locator(ticker.AutoLocator())
        plt.grid(True)
        plt.legend()
        plt.yscale('log')
        plt.tick_params(top=True, right=True, direction='in')

#####################################

def plot_gd(result_dict, output_dir, value_dicts):
    true_values = {'a': 1, 'b': 100}
    output_dir = os.path.join(output_dir, 'gradiend_descent')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gd.write_history(result_dict, output_dir)
    gd.contourplot(result_dict, true_values, value_dicts, output_dir)
    gd.angle_plot(result_dict, output_dir)
    gd.step_plot(result_dict, output_dir)
    rt.plot_progress(result_dict, true_values, output_dir)
    rt.plot_distance_history(result_dict, true_values, output_dir)
    rt.plot_fitness_history(result_dict, output_dir)
    rt.save_results(result_dict, output_dir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        choice = arguments['--choice']
        method = arguments['--method']
        output_dir = arguments['--output_dir']
        main(choice, method, output_dir)
    except docopt.DocoptExit as e:
        print(e)