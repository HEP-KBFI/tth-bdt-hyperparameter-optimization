import numpy as np
import sys
from tthAnalysis.bdtHyperparameterOptimization import global_functions as gf
import os
import shutil
dir_path = os.path.dirname(os.path.realpath(__file__))
resourcesDir = os.path.join(dir_path, 'resources', 'tmp')


def test_prepare_params_calc():
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
    expected = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
    }
    result = gf.prepare_params_calc(values)
    assert result == expected


def test_prepare_params_calc2():
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
    values_list = [
        values,
        values,
        values
    ]
    expected = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
    }
    expected_list = [
        expected,
        expected,
        expected
    ]
    result = gf.prepare_params_calc(values_list)
    assert result == expected_list


def test_initialize_values():
    value_dict1 = {
        'p_name': 'test1',
        'range_start': 0,
        'range_end': 10,
        'true_int': 'True'
    }
    value_dict2 = {
        'p_name': 'test2',
        'range_start': 0,
        'range_end': 10,
        'true_int': 'False'
    }
    value_dicts = [value_dict1, value_dict2]
    result = gf.initialize_values(value_dicts)
    assert result['test2'] >= 0 and result['test2'] <= 10
    assert isinstance(result['test1'], int)


def test_prepare_run_params():
    nthread = 28
    value_dict1 = {
        'p_name': 'test1',
        'range_start': 0,
        'range_end': 10,
        'true_int': 'True'
    }
    value_dict2 = {
        'p_name': 'test2',
        'range_start': 0,
        'range_end': 10,
        'true_int': 'False'
    }
    value_dicts = [value_dict1, value_dict2]
    sample_size = 3
    result = gf.prepare_run_params(
        nthread,
        value_dicts,
        sample_size
    )
    sum = 0
    for i in result:
        if isinstance(i['test1'], int):
            sum +=1
    assert len(result) == 3
    assert sum == 3


def test_calculate_f1_score():
    confusionMatrix = np.array([
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ])
    result = gf.calculate_f1_score(confusionMatrix)
    assert result[0] == 1
    assert result[1] == 1


def test_calculate_improvement_wSTDEV():
    parameter_dict1 = {
        'a': 1,
        'b': 10,
        'c': 5,
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': 2,
        'seed': 1
    }
    parameter_dict2 = {
        'a': 2,
        'b': 20,
        'c': 10,
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': 2,
        'seed': 1
    }
    parameter_dict3 = {
        'a': 3,
        'b': 30,
        'c': 15,
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': 2,
        'seed': 1
    }
    parameter_dicts = [
        parameter_dict1,
        parameter_dict2,
        parameter_dict3
    ]
    result = gf.calculate_improvement_wSTDEV(parameter_dicts)
    expected = np.sqrt(2/3)/2
    np.testing.assert_almost_equal(
        result,
        expected,
        7
    )


def test_values_to_list_dict():
    parameter_dict1 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dict2 = {'a': 4, 'b': 5, 'c': 6}
    parameter_dict3 = {'a': 7, 'b': 8, 'c': 9}
    parameter_dicts = [
        parameter_dict1,
        parameter_dict2,
        parameter_dict3
    ]
    keys = ['a', 'b', 'c']
    result = gf.values_to_list_dict(keys, parameter_dicts)
    expected = {
        'a': [1, 4, 7],
        'b': [2, 5, 8],
        'c': [3, 6, 9]
    }
    assert result == expected


def test_calculate_dict_mean_coeff_of_variation():
    list_dict = {
        'a': [1, 2, 3],
        'b': [10, 20, 30],
        'c': [5, 10, 15]
    }
    result = gf.calculate_dict_mean_coeff_of_variation(list_dict)
    expected = np.sqrt(2/3)/2
    np.testing.assert_almost_equal(
        result,
        expected,
        7
    )


def test_save_results():
    data_dict = {
        'dtrain': False,
        'dtest': False,
        'training_labels': [1, 2, 3, 4],
        'testing_labels': [1, 2, 3, 4]
    }
    best_parameters = {'x': 1, 'y': 2, 'z': 3}
    pred_train = [
        [0.9, 0.05, 0.03, 0.02],
        [0.1, 0.8, 0.05, 0.05],
        [0.1, 0.8, 0.05, 0.05],
        [0.1, 0.1, 0.1, 0.7]
    ]
    pred_test = [
        [0.9, 0.05, 0.03, 0.02],
        [0.1, 0.8, 0.05, 0.05],
        [0.1, 0.8, 0.05, 0.05],
        [0.1, 0.1, 0.1, 0.7]
    ]
    result_dict = {
        'data_dict': data_dict,
        'best_parameters': best_parameters,
        'pred_train': pred_train,
        'pred_test': pred_test,
        'best_fitness': 1,
        'avg_scores': [0.7, 0.8, 0.9, 1]
    }
    error = False
    try:
        gf.save_results(result_dict, resourcesDir)
    except:
        error = True
    assert error == False


def test_dummy_delete_files():
    if os.path.exists(resourcesDir):
        shutil.rmtree(resourcesDir)