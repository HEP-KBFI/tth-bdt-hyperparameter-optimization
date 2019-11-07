import numpy as np
import xgboost as xgb
from tthAnalysis.bdtHyperparameterOptimization import global_functions as gf
import docopt
import os
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
    value_dicts
):
    calc_dict = prepare_all_calculation(
        personal_bests, parameter_dicts, best_parameters)
    current_speeds = calculate_newSpeed(
        calc_dict, w, current_speeds)
    new_parameters = calculate_newValue(
        current_speeds, calc_dict['parameter_dicts'], nthread, value_dicts)
    return new_parameters


def prepare_all_calculation(
    personal_bests,
    parameter_dicts,
    best_parameters
):
    calc_dict = {}
    calc_dict['personal_bests'] = gf.prepare_params_calc(
        personal_bests)
    calc_dict['parameter_dicts'] = gf.prepare_params_calc(
        parameter_dicts)
    calc_dict['best_parameters'] = gf.prepare_params_calc(
        best_parameters)
    return calc_dict


def calculate_personal_bests(
    fitnesses,
    best_fitnesses,
    parameter_dicts,
    personal_bests
):
    new_dicts = []
    for fitness, best_fitness, parameters, personal_best in zip(
        fitnesses, best_fitnesses, parameter_dicts, personal_bests):
        if fitness > best_fitness:
            new_dicts.append(parameters)
        else:
            new_dicts.append(personal_best)
    return new_dicts


def calculate_newValue(
    current_speeds,
    parameter_dicts,
    nthread,
    value_dicts
):
    new_values = []
    params = {
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': nthread,
        'seed': 1
    }
    for current_speed, parameter_dict in zip(current_speeds, parameter_dicts):
        new_value = {}
        i = 0
        for speed, key in zip(current_speed, parameter_dict):
            if key == 'num_boost_round' or key == 'max_depth':
                new_value[key] = int(np.ceil(parameter_dict[key] + speed))
            else:
                new_value[key] = parameter_dict[key] + speed
            if new_value[key] < value_dicts[i]['range_start']:
                new_value[key] = value_dicts[i]['range_start']
            i += 1
        new_value.update(params)
        new_values.append(new_value)
    return new_values


def calculate_newSpeed(
    calc_dict,
    w,
    current_speeds,
    c1=1,
    c2=1
):
    new_speeds = []
    i = 0
    for pb, current in zip(
        calc_dict['personal_bests'],
        calc_dict['parameter_dicts']
    ):
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
                calc_dict['best_parameters'][key] - current[key]
            )
            cognitive_array.append(cognitive_component)
            social_array.append(social_component)
        cognitive_array = np.array(cognitive_array)
        social_array = np.array(social_array)
        new_speed = (
            w * inertia
            + c1 * r1 * cognitive_array
            + c2 * r2 * social_array
        )
        new_speeds.append(new_speed)
        i = i + 1
    return new_speeds


def read_weights(value_dicts, mainDir):
    path = os.path.join(mainDir, 'PSO', 'weights_runParameters.json')
    param_dict = gf.read_parameters(path)[0]
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
