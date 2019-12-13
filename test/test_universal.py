from __future__ import division
import numpy as np
import sys
import os
import shutil
from tthAnalysis.bdtHyperparameterOptimization import universal
dir_path = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(dir_path, 'resources', 'tmp')


def test_create_pairs():
    matrix = [[3, 7, 7], [4, 5, 6], [2, 9, 8]]
    expected = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]
    result = universal.create_pairs(matrix)
    assert result == expected


def test_normalization():
    elems = (4, 6)
    expected = (0.4, 0.6)
    result = universal.normalization(elems)
    assert result == expected


def test_create_mask():
    true_labels = [1, 0, 5, 4, 2]
    pair = (1, 3)
    expected = [True, False, False, False, False]
    result = universal.create_mask(true_labels, pair)
    assert result == expected


def test_get_values():
    matrix = [[3, 7, 7], [4, 5, 6], [2, 9, 8]]
    pair = (0, 2)
    expected = [
        (0.3, 0.7),
        (0.4, 0.6),
        (0.2, 0.8)
    ]
    result = universal.get_values(matrix, pair)
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
    result = universal.choose_values(matrix, pair, true_labels)
    print(result[1])
    assert (result[0] == labelsOut).all()
    assert (result[1] == elemsOut).all()


def test_plot_costfunction():
    avg_scores = [0.9, 0.95, 0.99, 1]
    error = False
    try:
        universal.plot_costfunction(avg_scores, resources_dir)
    except:
        error = True
    assert error == False


def test_dummy_delete_files():
    if os.path.exists(resources_dir):
        shutil.rmtree(resources_dir)



def test_calculate_f1_score():
    confusionMatrix = np.array([
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ])
    result = universal.calculate_f1_score(confusionMatrix)
    assert result[0] == 1
    assert result[1] == 1


def test_calculate_compactness():
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
    result = universal.calculate_compactness(parameter_dicts)
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
    result = universal.values_to_list_dict(keys, parameter_dicts)
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
    result = universal.calculate_dict_mean_coeff_of_variation(list_dict)
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
        universal.save_results(result_dict, resources_dir)
    except:
        error = True
    assert error == False


def test_read_parameters():
    path_to_test_file = os.path.join(
        dir_path, 'resources', 'best_parameters.json')
    result = universal.read_parameters(path_to_test_file)
    expected = [
        {'a': 1, 'b': 2, 'c': 3},
        {'stuff': 1}
    ]
    assert result == expected


def test_best_to_file():
    best_values = {'a': 1, 'b': 2, 'c': 3}
    assessment = {'g': 3, 'h': 4}
    good = True
    try:
        universal.best_to_file(best_values, resources_dir, assessment)
    except:
        good = False
    assert good


def test_calculate_d_score():
    pred_test = [
        [0.5, 0.4, 0.1],
        [0.5, 0.4, 0.1],
        [0.5, 0.4, 0.1],
        [0.5, 0.4, 0.1]
    ]
    pred_train = [
        [0.5, 0.4, 0.1],
        [0.5, 0.4, 0.1],
        [0.4, 0.5, 0.1],
        [0.5, 0.4, 0.1],
    ]
    data_dict = {
        'testing_labels': [0, 1, 0, 0],
        'training_labels': [0, 0, 1, 0]
    }
    d_score = universal.calculate_d_score(pred_train, pred_test, data_dict)
    expected = 0.75
    assert d_score == expected


def test_score():
    train_score = 1
    test_score = 1
    d_score = universal.score(train_score, test_score)
    assert d_score == 1


def test_calculate_d_roc():
    train_auc = 0.5
    test_auc = 0.5
    kappa = 0
    d_roc = universal.calculate_d_roc(train_auc, test_auc, kappa)
    expected = 0.5
    assert d_roc == expected


def test_calculate_conf_matrix():
    pred_train = [0, 1]
    pred_test = [0, 0]
    data_dict = {
        'training_labels': [0, 0],
        'testing_labels': [0, 1]
    }
    train_conf, test_conf = universal.calculate_conf_matrix(
        pred_train, pred_test, data_dict)
    expected1 = [[1, 0], [1, 0]]
    expected2 = [[1, 0], [1, 0]]
    assert (expected1 == train_conf).all()
    assert (expected2 == test_conf).all()


def test_get_most_probable():
    pred_train = [
        [0.3, 0.4, 0.3],
        [0.4, 0.5, 0.1],
        [0.8, 0.15, 0.05]
    ]
    pred_test = [
        [0.1, 0.4, 0.5],
        [0.5, 0.2, 0.3],
        [0.6, 0.2, 0.2]
    ]
    expected_train = [1, 1, 0]
    expected_test = [2, 0, 0]
    result = universal.get_most_probable(pred_train, pred_test)
    assert expected_test == result[1]
    assert expected_train == result[0]


def test_main_f1_calculate():
    pred_train = [0, 0]
    pred_test = [0, 0]
    data_dict = {
        'training_labels': [0, 1],
        'testing_labels': [0, 0]
    }
    result = universal.main_f1_calculate(pred_train, pred_test, data_dict)
    np.testing.assert_almost_equal(
        result['Train_F1'],
        2/3,
        6)
    np.testing.assert_almost_equal(
        result['Train_G'],
        np.sqrt(0.5),
        6
    )


def test_dummy_delete_files():
    if os.path.exists(resources_dir):
        shutil.rmtree(resources_dir)