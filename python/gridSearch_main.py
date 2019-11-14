from __future__ import division
from itertools import product
import numpy as np


def create_all_combinations(nr_parameters, grid_size):
    combinations = list(product(
        range(grid_size),
        repeat=nr_parameters)
    )
    return combinations


def initialize_values(parameters, grid_size):
    parameter_dicts = []
    combinations = create_all_combinations(len(parameters), grid_size)
    for iterations in combinations:
        parameter_dict = single_paramSet(parameters, iterations, grid_size)
        parameter_dicts.append(parameter_dict)
    return parameter_dicts


def single_paramSet(parameters, iterations, grid_size):
    parameter_dict = {}
    for param, iteration in zip(parameters, iterations):
        key = param['p_name']
        range_size = param['range_end'] - param['range_start']
        if grid_size == 1:
            value = (param['range_end'] + param['range_start']) / 2
        else:
            step_size = range_size / (grid_size - 1)
            value = param['range_start'] + (iteration * step_size)
        parameter_dict[key] = value
        if param['true_int'] == 'True':
            parameter_dict[key] = int(np.ceil(value))
    return parameter_dict


def perform_gridSearch(
    parameters,
    grid_size,
    nthread,
    calculateFitness,
    data_dict
):
    parameter_dicts = initialize_values(parameters, grid_size)
    print(':::::: Calculating fitnesses ::::::')
    fitnesses, pred_trains, pred_tests = calculateFitness(
        parameter_dicts, data_dict, nthread
    )
    index =  np.argmax(fitnesses)
    result_dict = {
        'best_parameters': parameter_dicts[index],
        'pred_train': pred_trains[index],
        'pred_test': pred_tests[index],
        'data_dict': data_dict
    }
    return result_dict
