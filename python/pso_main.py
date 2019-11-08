import numpy as np
import xgboost as xgb
# from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import prepare_params_calc
from tthAnalysis.bdtHyperparameterOptimization.universal import read_parameters
from tthAnalysis.bdtHyperparameterOptimization.universal import calculate_improvement_wSTDEV
# from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import prepare_run_params # also from XGBoost
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
    value_dicts,
    c1,
    c2
):
    current_speeds = calculate_newSpeed(
        personal_bests, param_dicts, best_parameters,
        w, current_speeds, c1, c2
    )
    new_parameters = calculate_newValue(
        current_speeds, parameter_dicts, nthread, value_dicts)
    return new_parameters

# shouldn't be xgb dependant here
# def prepare_all_calculation(
#     personal_bests,
#     parameter_dicts,
#     best_parameters
# ):
#     calc_dict = {}
#     calc_dict['personal_bests'] = prepare_params_calc(
#         personal_bests)
#     calc_dict['parameter_dicts'] = prepare_params_calc(
#         parameter_dicts)
#     calc_dict['best_parameters'] = prepare_params_calc(
#         best_parameters)
#     return calc_dict


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
        'silent': 1,
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
    personal_bests,
    param_dicts,
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
            + c1 * r1 * cognitive_array
            + c2 * r2 * social_array
        )
        new_speeds.append(new_speed)
        i = i + 1
    return new_speeds


def read_weights(value_dicts, mainDir):
    path = os.path.join(mainDir, 'weights_runParameters.json')
    param_dict = read_parameters(path)[0]
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


def run_pso(
    sample_dir,
    param_file,
    nthread,
    sample_size,
    w_init,
    w_fin,
    c1,
    c2,
    iterations,
    data_dict,
    value_dicts,
    calculate_fitnesses,
    number_parameters
):
    parameter_dicts = prepare_run_params( # liigutada algusesse.
        nthread, value_dicts, sample_size)
    w = w_init
    w_step = (w_fin - w_init)/iterations
    new_parameters = parameter_dicts
    personal_bests = {}
    # improvements = []
    # improvement = 1
    compactness_threshold = 0.1
    compactness = calculate_improvement_wSTDEV(parameter_dicts)
    i = 1
    print(":::::::: Initializing :::::::::")
    fitnesses, pred_trains, pred_tests = calculate_fitnesses(
        parameter_dicts, data_dict)
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
    current_speeds = np.zeros((sample_size, number_parameters))
    while i <= iterations and compactness_threshold < compactness:
        print("::::::: Iteration: ", i, " ::::::::")
        print(" --- Compactness: ", compactness, " ---")
        parameter_dicts = new_parameters
        fitnesses, pred_trains, pred_tests = calculate_fitnesses(
            parameter_dicts, data_dict)
        best_fitnesses = find_bestFitnesses(fitnesses, best_fitnesses)
        personal_bests = calculate_personal_bests(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests)
        new_parameters = prepare_newDay(
            personal_bests, parameter_dicts,
            result_dict['best_parameters'],
            current_speeds, w, nthread, value_dicts,
            c1, c2
        )
        index = np.argmax(fitnesses)
        if result_dict['best_fitness'] < max(fitnesses):
            result_dict['best_parameters'] = parameter_dicts[index]
            result_dict['pred_train'] = pred_trains[index]
            result_dict['pred_test'] = pred_tests[index]
            result_dict['best_fitness'] = max(fitnesses)
        avg_scores = np.mean(fitnesses)
        result_dict['avg_scores'].append(avg_scores)
        compactness = calculate_improvement_wSTDEV(parameter_dicts)
        # improvements, improvement = gf.calculate_improvement_wAVG(
        #     result_dict['avg_scores'],
        #     improvements,
        #     threshold
        # )
        w += w_step
        i += 1
    return result_dict