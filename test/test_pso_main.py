from __future__ import division
import numpy as np
import sys
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
import os
import shutil
import urllib
import gzip
from pathlib import Path
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import universal
dir_path = os.path.dirname(os.path.realpath(__file__))
hyper_path = str(Path(dir_path).parent)
resources_dir = os.path.join(dir_path, 'resources')
tmp_folder = os.path.join(resources_dir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)


main_url = 'http://yann.lecun.com/exdb/mnist/'
train_images = 'train-images-idx3-ubyte'
train_labels = 'train-labels-idx1-ubyte'
test_images = 't10k-images-idx3-ubyte'
test_labels = 't10k-labels-idx1-ubyte'
file_list = [train_labels, train_images, test_labels, test_images]
sample_dir = os.path.join(tmp_folder, 'samples_mnist')
nthread = 2
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
for file in file_list:
    file_loc = os.path.join(sample_dir, file)
    file_url = os.path.join(main_url, file + '.gz')
    urllib.urlretrieve(file_url, file_loc + '.gz')
    with gzip.open(file_loc + '.gz', 'rb') as f_in:
        with open(file_loc, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
data_dict = mf.create_datasets(sample_dir, 16)


value_dicts = [
    {
        'p_name': 'num_boost_round',
        'range_start': 1,
        'range_end': 500,
        'true_int': 1,
        'group_nr': 1,
        'true_corr': 0
    },
    {
        'p_name': 'learning_rate',
        'range_start': 0,
        'range_end': 0.3,
        'true_int': 0,
        'group_nr': 1,
        'true_corr': 0
    },
    {
        'p_name': 'max_depth',
        'range_start': 1,
        'range_end': 10,
        'true_int': 1,
        'group_nr': 2,
        'true_corr': 0
    },
    {
        'p_name': 'gamma',
        'range_start': 0,
        'range_end': 5,
        'true_int': 0,
        'group_nr': 2,
        'true_corr': 0
    },
    {
        'p_name': 'min_child_weight',
        'range_start': 0,
        'range_end': 500,
        'true_int': 0,
        'group_nr': 3,
        'true_corr': 0
    },
    {
        'p_name': 'subsample',
        'range_start': 0.8,
        'range_end': 1,
        'true_int': 0,
        'group_nr': 4,
        'true_corr': 0
    },
    {
        'p_name': 'colsample_bytree',
        'range_start': 0.3,
        'range_end': 1,
        'true_int': 0,
        'group_nr': 5,
        'true_corr': 0
    }
]


def test_find_best_fitness():
    fitnesses = [0.5, 0.7, 0.1, 0.2]
    best_fitnesses = [0.4, 0.8, 0.7, 0.3]
    expected = [0.5, 0.8, 0.7, 0.3]
    result = pm.find_best_fitness(fitnesses, best_fitnesses)
    assert result == expected


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


def test_calculate_new_speed():
    weight_dict = {
        'w': 2,
        'c1': 2,
        'c2': 2
    }
    current_speed = {'a': 1, 'b': 1, 'c': 1}
    current_speeds = [
        current_speed,
        current_speed,
        current_speed
    ]
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
    result = pm.calculate_new_speed(
        personal_bests,
        parameter_dicts,
        best_parameters,
        current_speeds,
        weight_dict
    )
    assert result[0]['a'] >= 2 and result[0]['a'] <= 20
    assert result[1]['b'] >= 2 and result[1]['b'] <= 14
    assert result[2]['c'] >= 0 and result[2]['c'] <= 10


def test_calculate_new_speed2():
    weight_dict = {
        'w': 2,
        'c1': 2,
        'c2': 2
    }
    values = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
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
        result = pm.calculate_new_speed(
            pb_list,
            current_values,
            best_params,
            current_speeds,
            weight_dict
        )
    except TypeError:
        error = True
    assert error == True


def test_calculate_new_position():
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
    values = {
        'num_boost_round': 1,
        'learning_rate': 1,
        'max_depth': 1,
        'gamma': 1,
        'min_child_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
    }
    current_speed = {
        'num_boost_round': 1,
        'learning_rate': 1,
        'max_depth': 1,
        'gamma': 1,
        'min_child_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
    }
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
    result = pm.calculate_new_position(
        current_speeds, parameter_dicts, value_dicts)
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


def test_check_numeric():
    variables1 = [0, 9, 0.99, 'a']
    variables2 = [0.99, 1/3, 0, 100, 1e3]
    result1 = pm.check_numeric(variables1)
    result2 = pm.check_numeric(variables2)
    assert result1
    assert not result2


def test_initialize_speeds():
    parameter_dicts = [
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 3, 'b': 2, 'c': 1}
    ]
    speeds = pm.initialize_speeds(parameter_dicts)
    expected = [
        {'a': 0, 'b': 0, 'c': 0},
        {'a': 0, 'b': 0, 'c': 0}
    ]
    assert speeds == expected


def test_get_weight_step():
    pso_settings = {'w_init': 1, 'w_fin': 0, 'iterations': 10}
    inertial_weight, inertial_weight_step = pm.get_weight_step(
        pso_settings)
    assert inertial_weight == 1
    assert inertial_weight_step == -0.1


def test_track_best_scores():
    feature_importances = [
        {'f1': 1, 'f2': 2},
        {'f1': 2, 'f2': 3},
        {'f1': 0.1, 'f2': 4}
    ]
    score_dicts = [
        {
            'g_score': 1,
            'f1_score': 1,
            'd_score': 1,
            'test_auc': 1,
            'train_auc': 1
        },
        {
            'g_score': 0.5,
            'f1_score': 0.5,
            'd_score': 0.5,
            'test_auc': 0.5,
            'train_auc': 0.5
        },
        {
            'g_score': 0.6,
            'f1_score': 0.6,
            'd_score': 0.6,
            'test_auc': 0.6,
            'train_auc': 0.6
        }
    ]
    keys = ['g_score', 'f1_score', 'd_score', 'test_auc', 'train_auc']
    parameter_dicts = [
        {'foo': 1, 'bar': 2},
        {'foo': 3, 'bar': 2},
        {'foo': 2, 'bar': 1}
    ]
    result_dict = {
        'best_g_score': 0.6,
        'best_f1_score': 0.6,
        'best_d_score': 0.6,
        'best_test_auc': 0.6,
        'best_train_auc': 0.6,
        'avg_scores': [1, 2],
        'compactnesses': [0.2, 0.3],
        'best_fitnesses': [0.8, 0.9]
    }
    fitnesses = [1, 0.5, 0.6]
    compactness = 0.1
    pred_trains = [
        [1, 2, 3, 4, 5],
        [2, 2, 3, 4, 5],
        [3, 2, 3, 4, 5]
    ]
    pred_tests = [
        [1, 2, 3, 4, 5],
        [2, 2, 3, 4, 5],
        [3, 2, 3, 4, 5]
    ]
    result_dict1 = pm.track_best_scores(
        feature_importances,
        parameter_dicts,
        keys,
        score_dicts,
        result_dict,
        fitnesses,
        compactness,
        pred_trains,
        pred_tests,
        new_bests=True,
        initialize_lists=False,
        append_lists=False
    )
    expected ={
        'best_g_score': 1,
        'best_f1_score': 1,
        'best_d_score': 1,
        'best_test_auc': 1,
        'best_train_auc': 1,
        'avg_scores': [1, 2],
        'compactnesses': [0.2, 0.3],
        'best_fitnesses': [0.8, 0.9],
        'feature_importances': {'f1': 1, 'f2': 2},
        'best_parameters': {'foo': 1, 'bar': 2},
        'best_fitness': 1,
        'pred_train': [1, 2, 3, 4, 5],
        'pred_test': [1, 2, 3, 4, 5]
    }
    assert result_dict1 == expected


def test_track_best_scores2():
    feature_importances = [
        {'f1': 1, 'f2': 2},
        {'f1': 2, 'f2': 3},
        {'f1': 0.1, 'f2': 4}
    ]
    score_dicts = [
        {
            'g_score': 1,
            'f1_score': 1,
            'd_score': 1,
            'test_auc': 1,
            'train_auc': 1
        },
        {
            'g_score': 0.5,
            'f1_score': 0.5,
            'd_score': 0.5,
            'test_auc': 0.5,
            'train_auc': 0.5
        },
        {
            'g_score': 0.6,
            'f1_score': 0.6,
            'd_score': 0.6,
            'test_auc': 0.6,
            'train_auc': 0.6
        }
    ]
    keys = ['g_score', 'f1_score', 'd_score', 'test_auc', 'train_auc']
    parameter_dicts = [
        {'foo': 1, 'bar': 2},
        {'foo': 3, 'bar': 2},
        {'foo': 2, 'bar': 1}
    ]
    result_dict = {
        'best_g_score': 0.6,
        'best_f1_score': 0.6,
        'best_d_score': 0.6,
        'best_test_auc': 0.6,
        'best_train_auc': 0.6
    }
    fitnesses = [1, 0.5, 0.6]
    compactness = 0.1
    pred_trains = [
        [1, 2, 3, 4, 5],
        [2, 2, 3, 4, 5],
        [3, 2, 3, 4, 5]
    ]
    pred_tests = [
        [1, 2, 3, 4, 5],
        [2, 2, 3, 4, 5],
        [3, 2, 3, 4, 5]
    ]
    result_dict1 = pm.track_best_scores(
        feature_importances,
        parameter_dicts,
        keys,
        score_dicts,
        result_dict,
        fitnesses,
        compactness,
        pred_trains,
        pred_tests,
        new_bests=True,
        initialize_lists=True,
        append_lists=True
    )
    expected ={
        'best_g_score': 1,
        'best_f1_score': 1,
        'best_d_score': 1,
        'best_test_auc': 1,
        'best_train_auc': 1,
        'avg_scores': np.mean([1, 0.5, 0.6]),
        'compactnesses': [0.1],
        'feature_importances': {'f1': 1, 'f2': 2},
        'best_parameters': {'foo': 1, 'bar': 2},
        'best_fitnesses': [1],
        'best_g_scores': [1],
        'best_f1_scores': [1],
        'best_d_scores': [1],
        'best_test_aucs': [1],
        'best_train_aucs': [1],
        'pred_train': [1, 2, 3, 4, 5],
        'pred_test': [1, 2, 3, 4, 5],
        'best_fitness': 1
    }
    assert result_dict1 == expected


def test_prepare_new_day():
    personal_bests = [
        {'foo': 1, 'bar': 2},
        {'foo': 2, 'bar': 2}
    ]
    parameter_dicts = [
        {'foo': 3, 'bar': 3},
        {'foo': 4, 'bar': 4}
    ]
    best_parameters = {'foo': 1, 'bar': 2}
    current_speeds = [
        {'foo': 0, 'bar': 0},
        {'foo': 0, 'bar': 0}
    ]
    value_dicts = [
    {'p_name': 'foo', 'range_start': 1, 'range_end': 500, 'true_int': 0},
    {'p_name': 'bar', 'range_start': 0, 'range_end': 0.3, 'true_int': 0}
    ]
    weight_dict = {'c1': 1, 'c2': 1, 'w': 1}
    error = False
    try:
        new_parameters, current_speeds = pm.prepare_new_day(
            personal_bests,
            parameter_dicts,
            best_parameters,
            current_speeds,
            value_dicts,
            weight_dict
        )
    except:
        error = True
    assert not error


def test_run_pso():
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
    param_file = os.path.join(
        main_dir,
        'data',
        'xgb_parameters.json'
    )
    value_dicts = universal.read_parameters(param_file)
    parameter_dicts = [
        {
            'num_boost_round': 71,
            'learning_rate': 0.07,
            'max_depth': 2,
            'gamma': 1.9,
            'min_child_weight': 18,
            'subsample': 0.9,
            'colsample_bytree': 0.8
        },
        {
            'num_boost_round': 72,
            'learning_rate': 0.17,
            'max_depth': 3,
            'gamma': 1.9,
            'min_child_weight': 18,
            'subsample': 0.9,
            'colsample_bytree': 0.8
        }
    ]
    global_settings_path = os.path.join(
        main_dir, 'data', 'global_settings.json')
    test_global_settings_path = os.path.join(
        main_dir, 'test', 'resources', 'global_settings.json')
    pso_settings_path = os.path.join(
        main_dir, 'data', 'pso_settings.json')
    test_pso_settings_path = os.path.join(
        main_dir, 'test', 'resources', 'pso_settings.json')
    os.rename(global_settings_path, global_settings_path + '_')
    os.rename(pso_settings_path, pso_settings_path + '_')
    shutil.copy(test_global_settings_path, global_settings_path)
    shutil.copy(test_pso_settings_path, pso_settings_path)
    result_dict = pm.run_pso(
        data_dict, value_dicts, xt.ensemble_fitnesses, parameter_dicts)
    os.rename(global_settings_path + '_', global_settings_path)
    os.rename(pso_settings_path + '_', pso_settings_path)
    assert result_dict != None


def test_read_weights():
    result = pm.read_weights()
    assert len(result) == 7


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
