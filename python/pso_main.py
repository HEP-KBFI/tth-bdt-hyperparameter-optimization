from __future__ import division
import os
import numbers
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
np.random.seed(1)


def find_bestFitnesses(fitnesses, best_fitnesses):
    new_best_fitnesses = []
    for fitness, best_fitness in zip(fitnesses, best_fitnesses):
        if fitness > best_fitness:
            new_best_fitnesses.append(fitness)
        else:
            new_best_fitnesses.append(best_fitness)
    return new_best_fitnesses


def prepare_newDay(
        personal_bests,
        parameter_dicts,
        best_parameters,
        current_speeds,
        w,
        nthread,
        value_dicts,
        c1,
        c2
):
    current_speeds = calculate_newSpeed(
        personal_bests, parameter_dicts, best_parameters,
        w, current_speeds, c1, c2
    )
    new_parameters = calculate_newValue(
        current_speeds, parameter_dicts, nthread, value_dicts)
    return new_parameters, current_speeds


def checkNumeric(variables):
    for variable in variables:
        if not isinstance(variable, numbers.Number):
            return True
    return False


def calculate_personal_bests(
        fitnesses,
        best_fitnesses,
        parameter_dicts,
        personal_bests
):
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

# XGB specific
def calculate_newValue(
        current_speeds,
        parameter_dicts,
        nthread,
        value_dicts
):
    new_values = []
    for current_speed, parameter_dict in zip(current_speeds, parameter_dicts):
        new_value = {}
        for i, speed in enumerate(current_speed):
            key = value_dicts[i]['p_name']
            if key == 'num_boost_round' or key == 'max_depth':
                new_value[key] = int(np.ceil(parameter_dict[key] + speed))
            else:
                new_value[key] = parameter_dict[key] + speed
            if new_value[key] < value_dicts[i]['range_start']:
                new_value[key] = value_dicts[i]['range_start']
            elif new_value[key] > value_dicts[i]['range_end']:
                new_value[key] = value_dicts[i]['range_end']
        new_values.append(new_value)
    return new_values


def calculate_newSpeed(
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
    for pb, current in zip(personal_bests, parameter_dicts):
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        inertia = np.array(current_speeds[i])
        cognitive_array = []
        social_array = []
        for key in current:
            cognitive_component = (
                pb[key] - current[key]
            )
            social_component = (
                best_parameters[key] - current[key]
            )
            cognitive_array.append(cognitive_component)
            social_array.append(social_component)
        cognitive_array = np.array(cognitive_array)
        social_array = np.array(social_array)
        new_speed = (
            w * inertia
            + c1 * (r1 * cognitive_array)
            + c2 * (r2 * social_array)
        )
        new_speeds.append(new_speed)
        i = i + 1
    return new_speeds


def read_weights(value_dicts):
    param_dict = universal.read_settings('pso')
    weight_dict = {
        'w_init': [],
        'w_fin': [],
        'c1': [],
        'c2': [],
        'iterations': param_dict['iterations'],
        'sample_size': param_dict['sample_size']
    }
    normed_weights_dict = weight_normalization(param_dict)
    for xgbParameter in value_dicts:
        if xgbParameter['range_end'] <= 1:
            weight_dict['w_init'].append(normed_weights_dict['w_init'])
            weight_dict['w_fin'].append(normed_weights_dict['w_fin'])
            weight_dict['c1'].append(normed_weights_dict['c1'])
            weight_dict['c2'].append(normed_weights_dict['c2'])
        else:
            weight_dict['w_init'].append(param_dict['w_init'])
            weight_dict['w_fin'].append(param_dict['w_fin'])
            weight_dict['c1'].append(param_dict['c1'])
            weight_dict['c2'].append(param_dict['c2'])
    return weight_dict


def weight_normalization(param_dict):
    total_sum = param_dict['w_init'] + param_dict['c1'] + param_dict['c2']
    normed_weights_dict = {
        'w_init': param_dict['w_init']/total_sum,
        'w_fin': param_dict['w_fin']/total_sum,
        'c1': param_dict['c1']/total_sum,
        'c2': param_dict['c2']/total_sum
    }
    return normed_weights_dict


def get_weight_step(pso_settings):
    w = np.array = pso_settings['w_init']
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
    w, w_step = get_weight_step(pso_settings)
    new_parameters = parameter_dicts
    personal_bests = {}
    compactness = universal.calculate_improvement_wSTDEV(parameter_dicts)
    i = 1
    print(':::::::: Initializing :::::::::')
    fitnesses, pred_trains, pred_tests = calculate_fitnesses(
        parameter_dicts, data_dict, pso_settings['sample_size'],
        global_settings)
    index = np.argmax(fitnesses)
    pso_settings['iterations']
    pso_settings['compactness_threshold']
    number_parameters = get_number_parameters()
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
    current_speeds = np.zeros((pso_settings['sample_size'], number_parameters))
    while i <= iteratons and compactness_threshold < compactness:
        print('::::::: Iteration: '+ str(i) + ' ::::::::')
        print(' --- Compactness: ' + str(compactness) + ' ---')
        parameter_dicts = new_parameters
        fitnesses, pred_trains, pred_tests = calculate_fitnesses(
            parameter_dicts, data_dict, pso_settings['sample_size'],
            global_settings)
        best_fitnesses = find_bestFitnesses(fitnesses, best_fitnesses)
        personal_bests = calculate_personal_bests(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests)
        new_parameters, current_speeds = prepare_newDay(
            personal_bests, parameter_dicts,
            result_dict['best_parameters'],
            current_speeds, w, global_settings['nthread'], value_dicts,
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
        compactness = universal.calculate_improvement_wSTDEV(parameter_dicts)
        w += w_step
        i += 1
    return result_dict
