'''Functions that are necessary for PSO algorithm
'''
from __future__ import division
import os
import numbers
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
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
        w,
        value_dicts,
        c1,
        c2
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
    w : float
        Inertial weight
    value_dicts : 
    '''
    current_speeds = calculate_new_speed(
        personal_bests, parameter_dicts, best_parameters,
        w, current_speeds, c1, c2
    )
    new_parameters = calculate_new_position(
        current_speeds, parameter_dicts, value_dicts)
    return new_parameters, current_speeds


def checkNumeric(variables):
    '''Checks whether the variable is numberic

    Parameters:
    ----------
    variables : list

    Returns:
    -------
    decision : bool
        Decision whether the list of variables contains non-numeric values
    '''
    for variable in variables:
        if not isinstance(variable, numbers.Number):
            decision = True
        else:
            decision = False
    return decision


def calculate_personal_bests(
        fitnesses,
        best_fitnesses,
        parameter_dicts,
        personal_bests
):
    '''Find best parameter-set for each particle

    Paraneters:
    ----------
    fitnesses : list
        List of current iteration fitnesses for each particle
    best_fitnesses : list
        List of best fitnesses for each particle
    parameter_dicts : list of dicts
        Current parameters of the last iteration for each particle
    personal_bests : list of dicts
        Best parameters (with highest fitness) for each particle so far
    '''
    new_dicts = []
    for fitness, best_fitness, parameters, personal_best in zip(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests):
        nonNumeric = checkNumeric(
            [fitness, best_fitness])
        if nonNumeric:
            raise TypeError
        if fitness > best_fitness:
            new_dicts.append(parameters)
        else:
            new_dicts.append(personal_best)
    return new_dicts

# # XGB specific
# def calculate_new_position(
#         current_speeds,
#         parameter_dicts,
#         value_dicts
# ):
#     new_values = []
#     for current_speed, parameter_dict in zip(current_speeds, parameter_dicts):
#         new_value = {}
#         for i, speed in enumerate(current_speed):
#             key = value_dicts[i]['p_name']
#             if bool(value_dicts[i]['true_int']):
#                 new_value[key] = int(np.ceil(parameter_dict[key] + speed))
#             else:
#                 new_value[key] = parameter_dict[key] + speed
#             if new_value[key] < value_dicts[i]['range_start']:
#                 new_value[key] = value_dicts[i]['range_start']
#             elif new_value[key] > value_dicts[i]['range_end']:
#                 new_value[key] = value_dicts[i]['range_end']
#         new_values.append(new_value)
#     return new_values


def calculate_new_position(
        current_speeds,
        parameter_dicts,
        value_dicts
):
    new_values = []
    for current_speed, parameter_dict in zip(current_speeds, parameter_dicts):
        for parameter in value_dicts:
            key = parameter['p_name']
            if bool(parameter['true_int']):
                new_value[key] = int(np.ceil(
                    parameter_dict[key] + current_speed[key]))
            else:
                new_value[key] = parameter_dict[key] + current_speed[key]
        new_values.append(new_value)

# def calculate_new_speed(
#         personal_bests,
#         parameter_dicts,
#         best_parameters,
#         w,
#         current_speeds,
#         c1,
#         c2
# ):
#     new_speeds = []
#     i = 0
#     for pb, current in zip(personal_bests, parameter_dicts):
#         rand1 = np.random.uniform()
#         rand2 = np.random.uniform()
#         inertia = np.array(current_speeds[i])
#         cognitive_array = []
#         social_array = []
#         for key in current:
#             cognitive_component = (
#                 pb[key] - current[key]
#             )
#             social_component = (
#                 best_parameters[key] - current[key]
#             )
#             cognitive_array.append(cognitive_component)
#             social_array.append(social_component)
#         cognitive_array = np.array(cognitive_array)
#         social_array = np.array(social_array)
#         new_speed = (
#             w * inertia
#             + c1 * (rand1 * cognitive_array)
#             + c2 * (rand2 * social_array)
#         )
#         new_speeds.append(new_speed)
#         i = i + 1
#     return new_speeds


def calculate_new_speed(
        personal_bests,
        parameter_dicts,
        best_parameters,
        w,
        current_speeds,
        c1,
        c2
):
    new_speeds = []
    i = 0
    for personal, current, inertia in zip(
        personal_bests, parameter_dicts, current_speeds
    ):
        new_speed = {}
        rand1 = np.random.uniform()
        rand2 = np.random.uniform()
        for key in current:
            cognitive_component = c1 * (personal[key] - current[key])
            social_component = c2 * (best_parameters[key] - current[key])
            inertial_component = w * inertia[key]
            new_speed[key] = (
                cognitive_component
                + social_component
                + inertial_component
            )
        new_speeds.append(new_speed)
    return 


def initialize_speeds(parameter_dicts):
    speeds = []
    for parameter_dict in parameter_dicts:
        speed = {}
        for key in parameter_dict:
            speed[key] = 0
        speeds.append(speed)
    return speeds


def read_weights(value_dicts):
    pso_settings = universal.read_settings('pso')
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
    total_sum = pso_settings['w_init'] + pso_settings['c1'] + pso_settings['c2']
    normed_weights_dict = {
        'w_init': pso_settings['w_init']/total_sum,
        'w_fin': pso_settings['w_fin']/total_sum,
        'c1': pso_settings['c1']/total_sum,
        'c2': pso_settings['c2']/total_sum
    }
    return normed_weights_dict


def get_weight_step(pso_settings):
    w = np.array(pso_settings['w_init'])
    w_fin = np.array(pso_settings['w_fin'])
    w_init = np.array(pso_settings['w_init'])
    w_step = (w_fin - w_init)/pso_settings['iterations']
    return w, w_step


def get_number_parameters():
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    parameters_path = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters.json')
    with open(parameters_path, 'rt') as parameters_file:
        number_parameters = len(parameters_file.readlines())
    return number_parameters


def run_pso(
        global_settings,
        pso_settings,
        data_dict,
        value_dicts,
        calculate_fitnesses,
        parameter_dicts,
):
    print(':::::::: Initializing :::::::::')
    w, w_step = get_weight_step(pso_settings)
    iterations = pso_settings['iterations']
    compactness_threshold = pso_settings['compactness_threshold']
    # number_parameters = get_number_parameters()
    i = 1
    new_parameters = parameter_dicts
    personal_bests = {}
    compactness = universal.calculate_compactness(parameter_dicts)
    fitnesses, pred_trains, pred_tests = calculate_fitnesses(
        parameter_dicts, data_dict, global_settings)
    index = np.argmax(fitnesses)
    result_dict = {
        'data_dict': data_dict,
        'best_parameters': parameter_dicts[index],
        'pred_train': pred_trains[index],
        'pred_test': pred_tests[index],
        'best_fitness': max(fitnesses),
        'avg_scores': [np.mean(fitnesses)]
    }
    personal_bests = parameter_dicts
    best_fitnesses = fitnesses
    current_speeds = initialize_speeds(parameter_dicts)
    # current_speeds = np.zeros((pso_settings['sample_size'], number_parameters))
    while i <= iterations and compactness_threshold < compactness:
        print('::::::: Iteration: '+ str(i) + ' ::::::::')
        parameter_dicts = new_parameters
        compactness = universal.calculate_compactness(parameter_dicts)
        print(' --- Compactness: ' + str(compactness) + ' ---')
        fitnesses, pred_trains, pred_tests = calculate_fitnesses(
            parameter_dicts, data_dict, global_settings)
        best_fitnesses = find_best_fitness(fitnesses, best_fitnesses)
        personal_bests = calculate_personal_bests(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests)
        new_parameters, current_speeds = prepare_new_day(
            personal_bests, parameter_dicts,
            result_dict['best_parameters'],
            current_speeds, w, value_dicts,
            pso_settings['c1'], pso_settings['c2']
        )
        index = np.argmax(fitnesses)
        if result_dict['best_fitness'] < max(fitnesses):
            result_dict['best_parameters'] = parameter_dicts[index]
            result_dict['pred_train'] = pred_trains[index]
            result_dict['pred_test'] = pred_tests[index]
            result_dict['best_fitness'] = max(fitnesses)
        avg_scores = np.mean(fitnesses)
        result_dict['avg_scores'].append(avg_scores)
        w += w_step
        i += 1
    return result_dict
