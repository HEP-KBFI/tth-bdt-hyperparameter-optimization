from __future__ import division
import numpy as np
import sys
import os
import shutil
from tthAnalysis.bdtHyperparameterOptimization.universal import calculate_f1_score
from tthAnalysis.bdtHyperparameterOptimization.universal import calculate_improvement_wSTDEV
from tthAnalysis.bdtHyperparameterOptimization.universal import save_results
from tthAnalysis.bdtHyperparameterOptimization.universal import calculate_dict_mean_coeff_of_variation
from tthAnalysis.bdtHyperparameterOptimization.universal import create_pairs
from tthAnalysis.bdtHyperparameterOptimization.universal import normalization
from tthAnalysis.bdtHyperparameterOptimization.universal import create_mask
from tthAnalysis.bdtHyperparameterOptimization.universal import get_values
from tthAnalysis.bdtHyperparameterOptimization.universal import choose_values
from tthAnalysis.bdtHyperparameterOptimization.universal import plot_costFunction
from tthAnalysis.bdtHyperparameterOptimization.universal import values_to_list_dict
dir_path = os.path.dirname(os.path.realpath(__file__))
resourcesDir = os.path.join(dir_path, 'resources', 'tmp')


def test_create_pairs():
    matrix = [[3, 7, 7], [4, 5, 6], [2, 9, 8]]
    expected = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]
    result = create_pairs(matrix)
    assert result == expected


def test_normalization():
    elems = (4, 6)
    expected = (0.4, 0.6)
    result = normalization(elems)
    assert result == expected


def test_create_mask():
    true_labels = [1, 0, 5, 4, 2]
    pair = (1, 3)
    expected = [True, False, False, False, False]
    result = create_mask(true_labels, pair)
    assert result == expected


def test_get_values():
    matrix = [[3, 7, 7], [4, 5, 6], [2, 9, 8]]
    pair = (0, 2)
    expected = [
        (0.3, 0.7),
        (0.4, 0.6),
        (0.2, 0.8)
    ]
    result = get_values(matrix, pair)
    assert result == expected


def test_choose_values():
    matrix = [[3, 7, 7], [4, 5, 6], [2, 9, 8]]
    pair = (0, 2)
    true_labels = [1, 2, 2]
    labelsOut = np.array([2, 2])
    elemsOut = np.array([
        [0.4, 0.6],
        [0.2, 0.8]
    ])
    result = choose_values(matrix, pair, true_labels)
    print(result[1])
    assert (result[0] == labelsOut).all()
    assert (result[1] == elemsOut).all()


def test_plot_costFunction():
    avg_scores = [0.9, 0.95, 0.99, 1]
    error = False
    try:
        plot_costFunction(avg_scores, resourcesDir)
    except:
        error = True
    assert error == False


def test_dummy_delete_files():
    if os.path.exists(resourcesDir):
        shutil.rmtree(resourcesDir)



def test_calculate_f1_score():
    confusionMatrix = np.array([
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ])
    result = calculate_f1_score(confusionMatrix)
    assert result[0] == 1
    assert result[1] == 1


def test_calculate_improvement_wSTDEV():
    parameter_dict1 = {
        'a': 1,
        'b': 10,
        'c': 5
    }
    parameter_dict2 = {
        'a': 2,
        'b': 20,
        'c': 10
    }
    parameter_dict3 = {
        'a': 3,
        'b': 30,
        'c': 15
    }
    parameter_dicts = [
        parameter_dict1,
        parameter_dict2,
        parameter_dict3
    ]
    result = calculate_improvement_wSTDEV(parameter_dicts)
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
    result = values_to_list_dict(keys, parameter_dicts)
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
    result = calculate_dict_mean_coeff_of_variation(list_dict)
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
        save_results(result_dict, resourcesDir)
    except:
        error = True
    assert error == False


def test_dummy_delete_files():
    if os.path.exists(resourcesDir):
        shutil.rmtree(resourcesDir)