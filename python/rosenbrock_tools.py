from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import json



def parameter_evaluation(
        parameter_dict,
        a,
        b
):
    '''Evaluation of the Rosenbrock function to find the minimum. For Rosenbrock
    function the global minimum is as (x, y) = (a, a^2). Thus for our case where
    a = 1 and b = 100, min = (1, 1)

    Parameters:
    ----------
    parameter_dict : dict
        x and y values for each possible solution.

    Returns:
    -------
    score : dict
        Contains only the 'rosenbrock_score'

    '''
    f_x_y = (
        (a - parameter_dict['x'])**2
        + b*(parameter_dict['y']- parameter_dict['x']**2)**2
    )
    score = (-1)*f_x_y
    pred_train = []
    pred_test = []
    feature_importance = {}
    return score


def ensemble_fitness(
        parameter_dicts,
        true_values
):
    scores = []
    for parameter_dict in parameter_dicts:
        score = parameter_evaluation(
            parameter_dict,
            true_values['a'],
            true_values['b'])
        scores.append(score)
    return scores


def check_distance(true_values, best_parameters):
    diff_dict = {}
    diff_sqared_sum = 0
    for key in true_values:
        diff_dict[key] = true_values[key] - best_parameters[key]
        diff_sqared_sum += diff_dict[key]**2
    distance = np.sqrt(diff_sqared_sum)
    return distance


def run_pso(
        parameter_dicts,
        true_values,
        value_dicts,
        output_dir,
        global_settings
):
    '''Performs the whole particle swarm optimization

    Parameters:
    ----------
    global_settings : dict
        Global settings for the run.
    pso_settings : dict
        Particle swarm settings for the run
    parameter_dicts : list of dicts
        The parameter-sets of all particles.

    Returns:
    -------
    result_dict : dict
        Dictionary that contains the results like best_parameters,
        best_fitnesses, avg_scores, pred_train, pred_test, data_dict
    '''
    print(':::::::: Initializing :::::::::')
    output_dir = os.path.expandvars(global_settings['output_dir'])
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = universal.read_settings(settings_dir, 'global')
    pso_settings = pm.read_weights(settings_dir)
    inertial_weight, inertial_weight_step = pm.get_weight_step(pso_settings)
    iterations = pso_settings['iterations']
    i = 1
    new_parameters = parameter_dicts
    personal_bests = {}
    fitnesses = ensemble_fitness(parameter_dicts, true_values)
    result_dict = {}
    index = np.argmax(fitnesses)
    result_dict['best_parameters'] = parameter_dicts[index]
    result_dict['list_of_old_bests'] = [parameter_dicts[index]]
    personal_bests = parameter_dicts
    best_fitnesses = fitnesses
    current_speeds = pm.initialize_speeds(parameter_dicts)
    distance = check_distance(true_values, result_dict['best_parameters'])
    while i <= iterations or distance < 1e-3:
        parameter_dicts = new_parameters
        fitnesses = ensemble_fitness(parameter_dicts, true_values)
        best_fitnesses = pm.find_best_fitness(fitnesses, best_fitnesses)
        personal_bests = pm.calculate_personal_bests(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests)
        weight_dict = {
            'c1': pso_settings['c1'],
            'c2': pso_settings['c2'],
            'w': inertial_weight}
        new_parameters, current_speeds = pm.prepare_new_day(
            personal_bests, parameter_dicts,
            result_dict['best_parameters'],
            current_speeds, value_dicts,
            weight_dict
        )
        if result_dict['best_fitness'] < max(fitnesses):
            index = np.argmax(fitnesses)
            result_dict['best_fitness'] = max(fitnesses)
            result_dict['best_parameters'] = parameter_dicts[index]
        distance = check_distance(true_values, result_dict['best_parameters'])
        result_dict['list_of_old_bests'].append(result_dict['best_parameters'])
        inertial_weight += inertial_weight_step
        i += 1
    return result_dict


def flatten_dict_list(result_dict):
    flattened_dict = {}
    for key in result_dict['best_parameters']:
        flattened_dict[key] = []
        for old_best in result_dict['list_of_old_bests']:
            flattened_dict[key].append(old_best[key])
    return flattened_dict


def plot_progress(result_dict, true_values, output_dir):

    param_progress = flatten_dict_list(result_dict)
    for key in param_progress:
        plot_out = os.path.join(output_dir, key + '_process.png')
        x_values = np.arange(len(param_progress[key]))
        plt.plot(x_values, param_progress[key], label='Predicted value')
        plt.plot(x_values, true_values[key])
        plt.xlabel('Iteration number / #')
        plt.ylabel('Minima location')
        axis = plt.gca()
        axis.set_aspect('auto', adjustable='box')
        axis.xaxis.set_major_locator(ticker.AutoLocator())
        plt.grid(True)
        plt.legend()
        plt.tick_params(top=True, right=True, direction='in')
        plt.savefig(plot_out, bbox_inches='tight')
        plt.close('all')


def plot_distance_history(result_dict, true_values, output_dir):
    true_parameters = {'x': true_values['a'], 'y': true_values['a']**2}
    distances = []
    for old_best in result_dict['list_of_old_bests']:
        distance = check_distance(true_values, old_best)
        distances.append(distance)
    plot_out = os.path.join(output_dir, 'distance_from_minima.png')
    x_values = np.arange(len(param_progress[key]))
    plt.plot(x_values, param_progress[key], label='Predicted value')
    plt.plot(x_values, true_parameters[key])
    plt.xlabel('Iteration number / #')
    plt.ylabel('Distance from minimum')
    axis = plt.gca()
    axis.set_aspect('auto', adjustable='box')
    axis.xaxis.set_major_locator(ticker.AutoLocator())
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.tick_params(top=True, right=True, direction='in')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')


def save_results(result_dict, output_dir):
    best_parameters_path = os.path.join(output_dir, 'best_parameters.json')
    best_parameter_history_path = os.path.join(output_dir, 'history.json')
    with open(best_parameters_path, 'w') as file:
        json.dump(result_dict['best_parameters'], file)
    with open(best_parameter_history_path, 'w') as file:
        json.dump(result_dict['list_of_old_bests'])


def initialize_values(value_dicts):
    '''Initializes the parameters according to the value dict specifications

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized

    Returns:
    -------
    sample : list of dicts
        Parameter-set for a particle
    '''
    sample = {}
    for parameters in value_dicts:
         sample[str(parameters['p_name'])] = np.random.randint(
            low=parameters['range_start'],
            high=parameters['range_end']
        )
    return sample


def prepare_run_params(value_dicts, sample_size):
    ''' Creates parameter-sets for all particles (sample_size)

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized
    sample_size : int
        Number of particles to be created

    Returns:
    -------
    run_params : list of dicts
        List of parameter-sets for all particles
    '''
    run_params = []
    for i in range(sample_size):
        run_param = initialize_values(value_dicts)
        run_params.append(run_param)
    return run_params