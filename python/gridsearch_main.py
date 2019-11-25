'''Main functions to be used by gridsearch script
'''

from __future__ import division
from itertools import product
import numpy as np


def create_all_combinations(nr_parameters, grid_size):
    '''Creates all possible combinations of parameters

    Parameters:
    ----------
    nr_parameters : int
        Number of parameters that are in the grid
    grid_size : int
        Number of values to be added to the grid per parameter

    Returns:
    -------
    combinations : list
        List of all the combinations
    '''
    combinations = list(product(
        range(grid_size),
        repeat=nr_parameters))
    return combinations


def initialize_values(parameters, grid_size):
    '''Initializes the values of the grid

    Parameters:
    ----------
    parameters : list of dicts
        Each element of the list contains the necessary info for each
        parameter
    grid_size : int
        Number of values in the grid per parameter

    Returns:
    -------
    parameter_dicts : list of dicts
        List of dictionaries that constitute the whole grid and that
        cover the whole grid.
    '''
    parameter_dicts = []
    combinations = create_all_combinations(len(parameters), grid_size)
    for iterations in combinations:
        parameter_dict = single_paramset(parameters, iterations, grid_size)
        parameter_dicts.append(parameter_dict)
    return parameter_dicts


def single_paramset(parameters, iterations, grid_size):
    '''Creates parameter dict with appropriate parameters

    Parameters:
    ----------
    parameters : list of dicts
        Each element of the list contains the necessary info for each
        parameter
    iterations: list
        Combination
    grid_size : int
        Number of values in the grid per parameter
    '''
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


def perform_gridsearch(
        parameters,
        calculate_fitness,
        data_dict,
        grid_settings
):
    '''Performs grid search with given settings and parameters.

    Parameters:
    ----------
    parameters : list of dicts
        Each element of the list contains the necessary info for each
        parameter
    calculate_fitness : method
        Function for fitness calculation
    data_dict : dict
        Dictionary that contains data and the correct labels
    grid_settings: dict
        Dictionary that contains the necessary parameters for grid search

    Returns:
    -------
    result_dict : dict
        Dictionary that contains the data_dict, pred_train, pred_test and
        the best_parameters
    '''
    parameter_dicts = initialize_values(
        parameters, grid_settings['grid_size'])
    print(':::::: Calculating fitnesses ::::::')
    fitnesses, pred_trains, pred_tests = calculate_fitness(
        parameter_dicts, data_dict,
        grid_settings['nthread'],
        grid_settings['num_class']
    )
    index = np.argmax(fitnesses)
    result_dict = {
        'best_parameters': parameter_dicts[index],
        'pred_train': pred_trains[index],
        'pred_test': pred_tests[index],
        'data_dict': data_dict
    }
    return result_dict
