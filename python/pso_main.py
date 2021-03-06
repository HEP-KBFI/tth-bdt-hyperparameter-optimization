'''Functions that are necessary for PSO algorithm
'''
from __future__ import division
import numbers
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
import os
np.random.seed(1)


def find_best_fitness(fitnesses, best_fitnesses):
    '''Compares the current best fitnesses with the current ones and
    substitutes one if it finds better

    Parameters:
    ----------
    fitnesses : list
        List of current iteration fitnesses
    best_fitnesses : list
        List of the best found fitnesses

    Returns:
    -------
    new_best_fitnesses : list
        List of best fitnesses taken into account the ones found current
        iteration
    '''
    new_best_fitnesses = []
    for fitness, best_fitness in zip(fitnesses, best_fitnesses):
        if fitness > best_fitness:
            new_best_fitnesses.append(fitness)
        else:
            new_best_fitnesses.append(best_fitness)
    return new_best_fitnesses


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
    current_speeds = calculate_new_speed(
        personal_bests, parameter_dicts, best_parameters,
        current_speeds, weight_dict
    )
    new_parameters = calculate_new_position(
        current_speeds, parameter_dicts, value_dicts)
    return new_parameters, current_speeds


def check_numeric(variables):
    '''Checks whether the variable is numeric

    Parameters:
    ----------
    variables : list

    Returns:
    -------
    decision : bool
        Decision whether the list of variables contains non-numeric values
    '''
    nr_nonnumeric = 0
    decision = False
    for variable in variables:
        if not isinstance(variable, numbers.Number):
            nr_nonnumeric += 1
    if nr_nonnumeric > 0:
        decision = True
    return decision


def calculate_personal_bests(
        fitnesses,
        best_fitnesses,
        parameter_dicts,
        personal_bests
):
    '''Find best parameter-set for each particle

    Parameters:
    ----------
    fitnesses : list
        List of current iteration fitnesses for each particle
    best_fitnesses : list
        List of best fitnesses for each particle
    parameter_dicts : list of dicts
        Current parameters of the last iteration for each particle
    personal_bests : list of dicts
        Best parameters (with highest fitness) for each particle so far

    Returns:
    -------
    new_dicts : list of dicts
        Personal best parameter-sets for each particle
    '''
    new_dicts = []
    for fitness, best_fitness, parameters, personal_best in zip(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests):
        non_numeric = check_numeric(
            [fitness, best_fitness])
        if non_numeric:
            raise TypeError
        if fitness > best_fitness:
            new_dicts.append(parameters)
        else:
            new_dicts.append(personal_best)
    return new_dicts


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
            if bool(parameter['true_int']):
                new_value[key] = int(np.ceil(
                    parameter_dict[key] + current_speed[key]))
            else:
                new_value[key] = parameter_dict[key] + current_speed[key]
            if parameter['range_start'] > new_value[key]:
                new_value[key] = parameter['range_start']
            elif parameter['range_end'] < new_value[key]:
                new_value[key] = parameter['range_end']
        new_values.append(new_value)
    return new_values


def calculate_new_speed(
        personal_bests,
        parameter_dicts,
        best_parameters,
        current_speeds,
        weight_dict
):
    '''Calculates the new speed in each parameter direction for all particles

    Parameters:
    ----------
    personal_bests : list of dicts
        Best parameters for each individual particle
    parameter_dicts : list of dicts
        Current iteration parameters for each particle
    current_speeds : list of dicts
        Speed in every parameter direction for each particle
    weight_dict : dict
        dictionary containing the normalized weights [w: inertial weight,
        c1: cognitive weight, c2: social weight]

    Returns:
    -------
    new_speeds : list of dicts
        The new speed of the particle in each parameter direction
    '''
    new_speeds = []
    for personal, current, inertia in zip(
            personal_bests, parameter_dicts, current_speeds
    ):
        new_speed = {}
        for key in current:
            rand1 = np.random.uniform()
            rand2 = np.random.uniform()
            cognitive_component = weight_dict['c1'] * rand1 * (
                personal[key] - current[key])
            social_component = weight_dict['c2'] * rand2 * (
                best_parameters[key] - current[key])
            inertial_component = weight_dict['w'] * inertia[key]
            new_speed[key] = (
                cognitive_component
                + social_component
                + inertial_component
            )
        new_speeds.append(new_speed)
    return new_speeds


def initialize_speeds(parameter_dicts):
    '''Initializes the speeds in the beginning to be 0

    Parameters:
    ----------
    parameter_dicts : list of dicts
        The parameter-sets of all particles.

    Returns:
    -------
    speeds : list of dicts
        Speeds of all particles in all parameter directions. All are 0
    '''
    speeds = []
    for parameter_dict in parameter_dicts:
        speed = {}
        for key in parameter_dict:
            speed[key] = 0
        speeds.append(speed)
    return speeds


def read_weights(settings_dir):
    ''' Reads the weights for different components and normalizes them

    Parameters:
    ----------
    None

    Returns:
    -------
    weight_dict : dict
        Contains all the weights for PSO
    '''
    pso_settings = universal.read_settings(settings_dir, 'pso')
    normed_weights_dict = weight_normalization(pso_settings)
    weight_dict = {
        'w_init': normed_weights_dict['w_init'],
        'w_fin': normed_weights_dict['w_fin'],
        'c1': normed_weights_dict['c1'],
        'c2': normed_weights_dict['c2'],
        'iterations': pso_settings['iterations'],
        'sample_size': pso_settings['sample_size'],
        'compactness_threshold': pso_settings['compactness_threshold']
    }
    return weight_dict


def weight_normalization(pso_settings):
    '''Normalizes the weights of the PSO

    Parameters:
    ----------
    pso_settings : dict
        Settings for the PSO

    Returns:
    -------
    normed_weights_dict : dict
        Normalized settings
    '''
    total_sum = pso_settings['w_init'] + pso_settings['c1'] + pso_settings['c2']
    normed_weights_dict = {
        'w_init': pso_settings['w_init']/total_sum,
        'w_fin': pso_settings['w_fin']/total_sum,
        'c1': pso_settings['c1']/total_sum,
        'c2': pso_settings['c2']/total_sum
    }
    return normed_weights_dict


def get_weight_step(pso_settings):
    '''Calculates the step size of the inertial weight

    Parameters:
    ----------
    pso_settings : dict
        PSO settings

    Returns:
    -------
    inertial_weight : float
        inertial weight
    inertial_weight_step : float
        Step size of the inertial weight
    '''
    inertial_weight = np.array(pso_settings['w_init'])
    inertial_weight_fin = np.array(pso_settings['w_fin'])
    inertial_weight_init = np.array(pso_settings['w_init'])
    inertial_weight_step = (
        inertial_weight_fin - inertial_weight_init)/pso_settings['iterations']
    return inertial_weight, inertial_weight_step


def track_best_scores(
        feature_importances,
        parameter_dicts,
        keys,
        score_dicts,
        result_dict,
        fitnesses,
        compactness,
        pred_trains,
        pred_tests,
        new_bests=False,
        initialize_lists=False,
        append_lists=False
):
    '''Tracks best scores to a dict

    Parameters:
    ----------
    keys : list
        list of keys to be added to the initial result_dict
    score_dicts : list of dicts
        list containing the scores for each particle
    result_dict : dict
        Dictionary containing the best scores
    [initialize=False]: bool
        Whether to initialize also the lists

    Returns:
    -------
    result_dict : dict
        Dictionary containing the best scores
    '''
    index = np.argmax(fitnesses)
    for key in keys:
        key_name = 'best_' + key
        list_key = key_name + 's'
        if new_bests:
            result_dict[key_name] = score_dicts[index][key]
        if initialize_lists:
            result_dict[list_key] = []
        if append_lists:
            result_dict[list_key].append(result_dict[key_name])
    if new_bests:
        result_dict['best_fitness'] = max(fitnesses)
        result_dict['pred_train'] = pred_trains[index]
        result_dict['pred_test'] = pred_tests[index]
        result_dict['best_parameters'] = parameter_dicts[index]
        result_dict['feature_importances'] = feature_importances[index]
    if initialize_lists:
        result_dict['avg_scores'] = []
        result_dict['compactnesses'] = []
        result_dict['best_fitnesses'] = []
    if append_lists:
        result_dict['avg_scores'].append(np.mean(fitnesses))
        result_dict['compactnesses'].append(compactness)
        result_dict['best_fitnesses'].append(result_dict['best_fitness'])
    return result_dict


def run_pso(
        data_dict,
        value_dicts,
        calculate_fitnesses,
        parameter_dicts,
        output_dir
):
    '''Performs the whole particle swarm optimization

    Parameters:
    ----------
    global_settings : dict
        Global settings for the run.
    pso_settings : dict
        Particle swarm settings for the run
    data_dict : dict
        Contains the data and labels
    value_dicts : list of dicts
        Info about every variable that is to be optimized
    calculate_fitnesses : method
        Function for fitness calculation
    parameter_dicts : list of dicts
        The parameter-sets of all particles.

    Returns:
    -------
    result_dict : dict
        Dictionary that contains the results like best_parameters,
        best_fitnesses, avg_scores, pred_train, pred_test, data_dict
    '''
    scoring_keys = ['g_score', 'f1_score', 'd_score', 'test_auc', 'train_auc']
    print(':::::::: Initializing :::::::::')
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = universal.read_settings(settings_dir, 'global')
    pso_settings = read_weights(settings_dir)
    inertial_weight, inertial_weight_step = get_weight_step(pso_settings)
    iterations = pso_settings['iterations']
    compactness_threshold = pso_settings['compactness_threshold']
    i = 1
    new_parameters = parameter_dicts
    personal_bests = {}
    compactness = universal.calculate_compactness(parameter_dicts)
    score_dicts, pred_trains, pred_tests, feature_importances = calculate_fitnesses(
        parameter_dicts, data_dict, global_settings)
    fitnesses = universal.fitness_to_list(
        score_dicts, fitness_key=global_settings['fitness_fn'])
    result_dict = {'data_dict': data_dict}
    result_dict = track_best_scores(
        feature_importances,
        parameter_dicts,
        scoring_keys,
        score_dicts,
        result_dict,
        fitnesses,
        compactness,
        pred_trains,
        pred_tests,
        new_bests=True,
        initialize_lists=True,
        append_lists=True
    )
    personal_bests = parameter_dicts
    best_fitnesses = fitnesses
    current_speeds = initialize_speeds(parameter_dicts)
    while i <= iterations and compactness_threshold < compactness:
        print('::::::: Iteration: '+ str(i) + ' ::::::::')
        parameter_dicts = new_parameters
        compactness = universal.calculate_compactness(parameter_dicts)
        print(' --- Compactness: ' + str(compactness) + ' ---')
        score_dicts, pred_trains, pred_tests, feature_importances = calculate_fitnesses(
            parameter_dicts, data_dict, global_settings)
        fitnesses = universal.fitness_to_list(
            score_dicts, fitness_key=global_settings['fitness_fn'])
        best_fitnesses = find_best_fitness(fitnesses, best_fitnesses)
        personal_bests = calculate_personal_bests(
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
        if result_dict['best_fitness'] < max(fitnesses):
            result_dict = track_best_scores(
                feature_importances,
                parameter_dicts,
                scoring_keys,
                score_dicts,
                result_dict,
                fitnesses,
                compactness,
                pred_trains,
                pred_tests,
                new_bests=True,
            )
        result_dict = track_best_scores(
            feature_importances,
            parameter_dicts,
            scoring_keys,
            score_dicts,
            result_dict,
            fitnesses,
            compactness,
            pred_trains,
            pred_tests,
            append_lists=True,
        )
        inertial_weight += inertial_weight_step
        i += 1
    return result_dict
