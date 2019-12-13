from __future__ import division
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import gridsearch_main as gm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt


def test_single_paramSet():
    grid_size = 3
    parameters = [
        {'p_name': 'first', 'range_start': 1, 'range_end': 3, 'true_int': 'True'},
        {'p_name': 'second', 'range_start': 5, 'range_end': 7, 'true_int': 'False'},
        {'p_name': 'third', 'range_start': 8, 'range_end': 10, 'true_int': 'True'}
    ]
    iterations = [0, 1, 0]
    expected = {
        'first': 1,
        'second': 6,
        'third': 8
    }
    result = gm.single_paramSet(parameters, iterations, grid_size)
    assert result == expected


def test_initialize_values1():
    grid_size = 1
    parameters = [
        {'p_name': 'first', 'range_start': 1, 'range_end': 3, 'true_int': 'True'},
        {'p_name': 'second', 'range_start': 5, 'range_end': 7, 'true_int': 'False'},
        {'p_name': 'third', 'range_start': 8, 'range_end': 10, 'true_int': 'True'}
    ]
    expected1 = [{
        'first': 2.,
        'second': 6.,
        'third': 9.
    }]
    result = gm.initialize_values(parameters, grid_size)
    assert result == expected1


def test_initialize_values2():
    grid_size = 2
    parameters = [
        {'p_name': 'first', 'range_start': 1, 'range_end': 2, 'true_int': 'True'},
        {'p_name': 'second', 'range_start': 5, 'range_end': 6, 'true_int': 'False'},
    ]
    expected1 = {
        'first': 1.,
        'second': 5.,
    }
    expected2 = {
        'first': 1.,
        'second': 6.,
    }
    expected3 = {
        'first': 2.,
        'second': 5.,
    }
    expected4 = {
        'first': 2.,
        'second': 6.,
    }
    expected_l = [expected1, expected2, expected3, expected4]
    result = gm.initialize_values(parameters, grid_size)
    assert result == expected_l


def test_create_all_combinations():
    nr_parameters = 3
    grid_size = 3
    result = gm.create_all_combinations(nr_parameters, grid_size)
    assert len(result) == 27
