from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import json
from mpl_toolkits import mplot3d


def parameter_evaluation(
        parameter_dict,
        a=1,
        b=100
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
    score = (
        (a - parameter_dict['x'])**2
        + b*(parameter_dict['y']- parameter_dict['x']**2)**2
    )
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
    true_parameters = {'x': true_values['a'], 'y': true_values['a']**2}
    diff_dict = {}
    diff_sqared_sum = 0
    for key in true_parameters:
        diff_dict[key] = true_parameters[key] - best_parameters[key]
        diff_sqared_sum += diff_dict[key]**2
    distance = np.sqrt(diff_sqared_sum)
    return distance


def run_pso(
        parameter_dicts,
        true_values,
        value_dicts,
        output_dir,
        global_settings,
        plot_pso_location=False
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
    pso_settings = universal.read_settings(settings_dir, 'pso')
    inertial_weight, inertial_weight_step = pm.get_weight_step(pso_settings)
    iterations = pso_settings['iterations']
    i = 0
    new_parameters = parameter_dicts
    personal_bests = {}
    fitnesses = ensemble_fitness(parameter_dicts, true_values)
    result_dict = {}
    index = np.argmin(fitnesses)
    result_dict['best_fitness'] = fitnesses[index]
    result_dict['best_parameters'] = parameter_dicts[index]
    result_dict['list_of_old_bests'] = [parameter_dicts[index]]
    result_dict['list_of_best_fitnesses'] = [fitnesses[index]]
    personal_bests = parameter_dicts
    best_fitnesses = fitnesses
    current_speeds = pm.initialize_speeds(parameter_dicts)
    distance = check_distance(true_values, result_dict['best_parameters'])
    print('::::::::::: Optimizing ::::::::::')
    while i <= iterations or distance < 1e-8:
        print('---- Iteration: ' + str(i) + '----')
        parameter_dicts = new_parameters
        if plot_pso_location and i % 500 == 0:
            plot_particle_swarm(
                parameter_dicts, true_values, i, output_dir, result_dict)
        fitnesses = ensemble_fitness(parameter_dicts, true_values)
        best_fitnesses = pm.find_best_fitness(fitnesses, best_fitnesses)
        personal_bests = pm.calculate_personal_bests(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests)
        weight_dict = {
            'c1': pso_settings['c1'],
            'c2': pso_settings['c2'],
            'w': inertial_weight}
        new_parameters, current_speeds = prepare_new_day(
            personal_bests, parameter_dicts,
            result_dict['best_parameters'],
            current_speeds, value_dicts,
            weight_dict
        )
        if result_dict['best_fitness'] > min(fitnesses):
            index = np.argmin(fitnesses)
            result_dict['best_fitness'] = min(fitnesses)
            result_dict['best_parameters'] = parameter_dicts[index]
        distance = check_distance(true_values, result_dict['best_parameters'])
        result_dict['list_of_old_bests'].append(result_dict['best_parameters'])
        result_dict['list_of_best_fitnesses'].append(result_dict['best_fitness'])
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


def plot_particle_swarm(
        parameter_dicts,
        true_values,
        iteration,
        output_dir,
        result_dict
):
    iteration_pic_path = os.path.join(output_dir, 'pso_iteration_pictures')
    if not os.path.exists(iteration_pic_path):
        os.makedirs(iteration_pic_path)
    true_parameters = {'x': true_values['a'], 'y': true_values['a']**2}
    plot_out = os.path.join(
        iteration_pic_path, 'iteration_' + str(iteration) + '.png')
    for parameter_dict in parameter_dicts:
        plt.plot(
            parameter_dict['x'],
            parameter_dict['y'],
            color='k',
            marker='o')
    # for old_best in result_dict['list_of_old_bests']:
    #     plt.plot(
    #         old_best['x'],
    #         old_best['y'],
    #         color='g',
    #         marker='o',
    #         label='Previous bests')
    # plt.plot(
    #     result_dict['list_of_old_bests'][-1]['x'],
    #     result_dict['list_of_old_bests'][-1]['y'],
    #     color='b',
    #     marker='o',
    #     label='Current best')
    plt.plot(
        true_parameters['x'],
        true_parameters['y'],
        color='r',
        marker='o',
        label='Global minimum')
    plt.ylim(-500, 500)
    plt.xlim(-500, 500)
    plt.grid(True)
    plt.title('Iteration ' + str(iteration))
    axis = plt.gca()
    axis.set_aspect('auto', adjustable='box')
    axis.xaxis.set_major_locator(ticker.AutoLocator())
    plt.tick_params(top=True, right=True, direction='in')
    plt.savefig(plot_out)
    plt.close('all')


def plot_progress(result_dict, true_values, output_dir):
    true_parameters = {'x': true_values['a'], 'y': true_values['a']**2}
    param_progress = flatten_dict_list(result_dict)
    for key in param_progress:
        plot_out = os.path.join(output_dir, key + '_process.png')
        x_values = np.arange(len(param_progress[key]))
        plt.plot(x_values, param_progress[key], label='Predicted value')
        plt.axhline(true_parameters[key], color='r', linestyle='-')
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
    x_values = np.arange(len(distances))
    plt.plot(x_values, distances, label='Predicted value')
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel('Iteration number / #')
    plt.ylabel('Distance from minimum')
    axis = plt.gca()
    axis.set_aspect('auto', adjustable='box')
    axis.xaxis.set_major_locator(ticker.AutoLocator())
    plt.grid(True)
    plt.legend()
    # plt.yscale('log')
    plt.tick_params(top=True, right=True, direction='in')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')


def plot_fitness_history(
        result_dict, output_dir, close=True, label='Best fitness'
):
    plot_out = os.path.join(output_dir, 'best_fitnesses.png')
    x_values = np.arange(len(result_dict['list_of_best_fitnesses']))
    plt.plot(
        x_values,
        result_dict['list_of_best_fitnesses'],
        label=label
    )
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel('Iteration number / #')
    plt.ylabel('Fitness')
    axis = plt.gca()
    axis.set_aspect('auto', adjustable='box')
    axis.xaxis.set_major_locator(ticker.AutoLocator())
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.tick_params(top=True, right=True, direction='in')
    if close:
        plt.savefig(plot_out, bbox_inches='tight')
        plt.close('all')


def save_results(result_dict, output_dir):
    best_parameters_path = os.path.join(output_dir, 'best_parameters.json')
    best_parameter_history_path = os.path.join(output_dir, 'history.json')
    with open(best_parameters_path, 'w') as file:
        json.dump(result_dict['best_parameters'], file)
    with open(best_parameter_history_path, 'w') as file:
        json.dump(result_dict['list_of_old_bests'], file)


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


def calculate_new_position(
        current_speeds,
        parameter_dicts,
        value_dicts
):
    '''Calculates the new parameters for the next iteration

    Parameters:
    ----------
    current_speeds : list of dicts
        Current speed in each parameter direction for each particle
    parameter_dicts : list of dicts
        Current parameter-sets of all particles
    value_dicts : list of dicts
        Info about every variable that is to be optimized

    Returns:
    -------
    new_values : list of dicts
        New parameters to be used in the next iteration
    '''
    new_values = []
    for current_speed, parameter_dict in zip(current_speeds, parameter_dicts):
        new_value = {}
        for parameter in value_dicts:
            key = parameter['p_name']
            minimum = parameter['range_start']
            maximum = parameter['range_end']
            new_value[key] = parameter_dict[key] + current_speed[key]
            if new_value[key] > maximum:
                new_value[key] = maximum
            elif new_value[key] < minimum:
                new_value[key] = minimum
        new_values.append(new_value)
    return new_values


def prepare_new_day(
        personal_bests,
        parameter_dicts,
        best_parameters,
        current_speeds,
        value_dicts,
        weight_dict
):
    '''Finds the new new parameters to find the fitness of

    Parameters:
    ----------
    personal_bests : list of dicts
        Best parameters for each individual particle
    parameter_dicts : list of dicts
        Current iteration parameters for each particle
    current_speeds : list of dicts
        Speed in every parameter direction for each particle
    value_dicts : list of dicts
        Info about every variable that is to be optimized
    weight_dict : dict
        dictionary containing the normalized weights [w: inertial weight,
        c1: cognitive weight, c2: social weight]

    Returns:
    -------
    new_parameters : list of dicts
        Parameter-sets that are used in the next iteration
    current_speeds : list of dicts
        New speed of each particle
    '''
    current_speeds = pm.calculate_new_speed(
        personal_bests, parameter_dicts, best_parameters,
        current_speeds, weight_dict
    )
    new_parameters = calculate_new_position(
        current_speeds, parameter_dicts, value_dicts)
    return new_parameters, current_speeds


def plot_2d_location_progress(result_dict, true_values, output_dir):
    plot_out = os.path.join(output_dir, '2d_progress.png')
    flattened_dict = flatten_dict_list(result_dict)
    plt.plot(
        flattened_dict['x'],
        flattened_dict['y'],
        label='Approximation'
    )
    plt.plot(
        true_values['a'],
        true_values['a']**2,
        marker='o',
        markersize=3,
        color="red",
        label='True minimum'
    )
    plt.xlabel('x position')
    plt.ylabel('y position')
    axis = plt.gca()
    axis.set_aspect('auto', adjustable='box')
    axis.xaxis.set_major_locator(ticker.AutoLocator())
    plt.grid(True)
    plt.yscale('symlog')
    plt.legend()
    plt.tick_params(top=True, right=True, direction='in')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')



def run_random(
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
    pso_settings = universal.read_settings(settings_dir, 'pso')
    iterations = pso_settings['iterations']
    i = 1
    new_parameters = parameter_dicts
    personal_bests = {}
    fitnesses = ensemble_fitness(parameter_dicts, true_values)
    result_dict = {}
    index = np.argmin(fitnesses)
    result_dict['best_fitness'] = fitnesses[index]
    result_dict['best_parameters'] = parameter_dicts[index]
    result_dict['list_of_old_bests'] = [parameter_dicts[index]]
    result_dict['list_of_best_fitnesses'] = [fitnesses[index]]
    personal_bests = parameter_dicts
    best_fitnesses = fitnesses
    print('::::::::::: Optimizing ::::::::::')
    while i <= iterations:
        print('---- Iteration: ' + str(i) + '----')
        parameter_dicts = prepare_run_params(
            value_dicts, pso_settings['sample_size'])
        fitnesses = ensemble_fitness(parameter_dicts, true_values)
        best_fitnesses = pm.find_best_fitness(fitnesses, best_fitnesses)
        if result_dict['best_fitness'] > min(fitnesses):
            index = np.argmin(fitnesses)
            result_dict['best_fitness'] = min(fitnesses)
            result_dict['best_parameters'] = parameter_dicts[index]
        distance = check_distance(true_values, result_dict['best_parameters'])
        result_dict['list_of_old_bests'].append(result_dict['best_parameters'])
        result_dict['list_of_best_fitnesses'].append(result_dict['best_fitness'])
        i += 1
    return result_dict


def plot_3d_contour(
        parameter_dicts,
        true_values,
        iteration,
        output_dir,
        result_dict,
        max_unit=500,
        count=100
):
    iteration_pic_path = os.path.join(output_dir, 'pso_3d_iteration_pictures')
    if not os.path.exists(iteration_pic_path):
        os.makedirs(iteration_pic_path)
    true_parameters = {'x': true_values['a'], 'y': true_values['a']**2}
    plot_out = os.path.join(
        iteration_pic_path, 'iteration_' + str(iteration) + '.png')
    X, Y, Z = create_meshgrid(max_unit, count)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(
        [true_parameters['x']],
        [true_parameters['y']],
        [parameter_evaluation(true_parameters)],
        marker='o',
        color='r'
    )
    ax.contour3D(X, Y, Z, 100, cmap='viridis')
    for parameter_dict in parameter_dicts:
        ax.plot3D(
            [parameter_dict['x']],
            [parameter_dict['y']],
            [parameter_evaluation(parameter_dict)],
            color='k',
            marker='o')
    plt.savefig(plot_out)
    plt.close('all')


def create_meshgrid(max_unit, count):
    x_values = np.linspace(-max_unit, max_unit, count)
    y_values = np.linspace(-max_unit, max_unit, count)
    X, Y = np.meshgrid(x_values, y_values)
    loc_2d = {'x': X, 'y': Y}
    Z = parameter_evaluation(loc_2d)
    return X, Y, Z


def plot_2d_contour(
        parameter_dicts,
        true_values,
        iteration,
        output_dir,
        result_dict,
        max_unit=500,
        count=100
):
    iteration_pic_path = os.path.join(output_dir, 'pso_2d_iteration_pictures')
    if not os.path.exists(iteration_pic_path):
        os.makedirs(iteration_pic_path)
    true_parameters = {'x': true_values['a'], 'y': true_values['a']**2}
    plot_out = os.path.join(
        iteration_pic_path, 'iteration_' + str(iteration) + '.png')
    X, Y, Z = create_meshgrid(max_unit, count)
    plt.contour(X, Y, Z, 500)
    plt.plot(true_parameters['x'], true_parameters['y'], marker='o', color='r')
    for parameter_dict in parameter_dicts:
        plt.plot(
            [parameter_dict['x']],
            [parameter_dict['y']],
            [parameter_evaluation(parameter_dict)],
            color='k',
            marker='o')
    plt.grid(True)
    plt.savefig(plot_out)
    plt.close('all')