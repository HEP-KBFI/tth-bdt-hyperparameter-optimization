from __future__ import division
import numpy as np
import sys
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
hyper_path = str(Path(dir_path).parent)


value_dicts = [
    {
        'p_name': 'num_boost_round',
        'range_start': 1,
        'range_end': 500,
        'true_int': 'True',
        'group_nr': 1,
        'true_corr': 'False'
    },
    {
        'p_name': 'learning_rate',
        'range_start': 0,
        'range_end': 0.3,
        'true_int': 'False',
        'group_nr': 1,
        'true_corr': 'False'
    },
    {
        'p_name': 'max_depth',
        'range_start': 1,
        'range_end': 10,
        'true_int': 'True',
        'group_nr': 2,
        'true_corr': 'False'
    },
    {
        'p_name': 'gamma',
        'range_start': 0,
        'range_end': 5,
        'true_int': 'False',
        'group_nr': 2,
        'true_corr': 'False'
    },
    {
        'p_name': 'min_child_weight',
        'range_start': 0,
        'range_end': 500,
        'true_int': 'False',
        'group_nr': 3,
        'true_corr': 'True'
    },
    {
        'p_name': 'subsample',
        'range_start': 0.8,
        'range_end': 1,
        'true_int': 'False',
        'group_nr': 4,
        'true_corr': 'True'
    },
    {
        'p_name': 'colsample_bytree',
        'range_start': 0.3,
        'range_end': 1,
        'true_int': 'False',
        'group_nr': 5,
        'true_corr': 'True'
    }
]


def test_calculate_personal_bests():
    fitnesses = [0.9, 1, 0.9]
    best_fitnesses = [0.8, 0.9, 1]
    parameter_dicts = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 3, 'b': 3, 'c': 3}
    ]
    personal_bests = [
        {'a': 9, 'b': 9, 'c': 9},
        {'a': 8, 'b': 8, 'c': 8},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    result = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    calculated_pb = pm.calculate_personal_bests(
        fitnesses,
        best_fitnesses,
        parameter_dicts,
        personal_bests
    )
    assert result == calculated_pb


def test_calculate_personal_bests2():
    fitnesses = ['a', 1, 0.9]
    best_fitnesses = [0.8, 0.9, 1]
    parameter_dicts = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 3, 'b': 3, 'c': 3}
    ]
    personal_bests = [
        {'a': 9, 'b': 9, 'c': 9},
        {'a': 8, 'b': 8, 'c': 8},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    result = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    error = False
    try:
        calculated_pb = pm.calculate_personal_bests(
            fitnesses,
            best_fitnesses,
            parameter_dicts,
            personal_bests
        )
    except TypeError:
        error = True
    assert error == True


def test_calculate_newSpeed():
    w = 2
    c1 = 2
    c2 = 2
    current_speeds = [1, 1, 1]
    parameter_dicts = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 3, 'b': 3, 'c': 3}
    ]
    personal_bests = [
        {'a': 9, 'b': 9, 'c': 9},
        {'a': 8, 'b': 8, 'c': 8},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    best_parameters = {'a': 2, 'b': 2, 'c': 2}
    result = pm.calculate_newSpeed(
        personal_bests,
        parameter_dicts,
        best_parameters,
        w,
        current_speeds,
        c1,
        c2
    )
    assert result[0][0] >= 2 and result[0][0] <= 20
    assert result[1][0] >= 2 and result[1][0] <= 14
    assert result[2][0] >= 0 and result[2][0] <= 10


def test_calculate_newSpeed2():
    w = 2
    c1 = 2
    c2 = 2
    values = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': 2,
        'seed': 1
    }
    current_speeds = [1, 1, 1]
    current_values = [
        values,
        values,
        values
    ]
    pb_list = [
        values,
        values,
        values
    ]
    best_params = values
    error = False
    try:
        result = pm.calculate_newSpeed(
            w,
            current_values,
            current_speeds,
            best_params,
            pb_list,
            c1,
            c2
        )
    except TypeError:
        error = True
    assert error == True


def test_calculate_newValue():
    parameter_dict = {
        'num_boost_round': 0,
        'learning_rate': 0,
        'max_depth': 0,
        'gamma': 0,
        'min_child_weight': 0,
        'subsample': 0,
        'colsample_bytree': 0,
    }
    parameter_dicts = [
        parameter_dict,
        parameter_dict,
        parameter_dict
    ]
    nthread = 28
    values = {
        'num_boost_round': 1,
        'learning_rate': 1,
        'max_depth': 1,
        'gamma': 1,
        'min_child_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
    }
    current_speed = [1, 1, 1, 1, 1, 1, 1]
    current_speeds = [
        current_speed,
        current_speed,
        current_speed
    ]
    expected = [
        values,
        values,
        values
    ]
    result = pm.calculate_newValue(
        current_speeds, parameter_dicts, nthread, value_dicts)
    assert result == expected


def test_weight_normalization():
    param_dict = {
        'iterations': 2,
        'sample_size': 3,
        'compactness_threshold': 0.1,
        'w_init': 0.9,
        'w_fin': 0.4,
        'c1': 2,
        'c2': 2
    }
    result = pm.weight_normalization(param_dict)
    np.testing.assert_almost_equal(
        result['w_init'],
        0.18,
        2
    )
    np.testing.assert_almost_equal(
        result['c1'],
        0.408,
        2
    )
    np.testing.assert_almost_equal(
        result['w_fin'],
        0.08,
        2
    )


def test_read_weights():
    weightPaht = os.path.join(hyper_path, 'data')
    result = pm.read_weights(value_dicts, weightPaht)
    assert len(result) == 6
    assert len(result['c1']) == len(value_dicts)