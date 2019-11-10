import numpy as np
from tthAnalysis.bdtHyperparameterOptimization.xgb_gridSearch import single_paramSet
from tthAnalysis.bdtHyperparameterOptimization.xgb_gridSearch import initialize_values
from tthAnalysis.bdtHyperparameterOptimization.xgb_gridSearch import param_update
from tthAnalysis.bdtHyperparameterOptimization.xgb_gridSearch import create_all_combinations


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
    result = single_paramSet(parameters, iterations, grid_size)
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
    result = initialize_values(parameters, grid_size)
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
    result = initialize_values(parameters, grid_size)
    assert result == expected_l


def test_param_update():
    nthread = 2
    dict1 = {
        'first': 1.,
        'second': 5.,
        'third': 8.,
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': nthread,
        'seed': 1,
    }
    dict2 = {
        'first': 2.,
        'second': 6.,
        'third': 9.,
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': nthread,
        'seed': 1,
    }
    expected_dicts = [dict1, dict2]
    in1 = {
        'first': 1.,
        'second': 5.,
        'third': 8.
    }
    in2 = {
        'first': 2.,
        'second': 6.,
        'third': 9.
    }
    in_d = [in1, in2]
    result = param_update(in_d, nthread)
    assert result == expected_dicts


def test_create_all_combinations():
    nr_parameters = 3
    grid_size = 3
    result = create_all_combinations(nr_parameters, grid_size)
    assert len(result) == 27