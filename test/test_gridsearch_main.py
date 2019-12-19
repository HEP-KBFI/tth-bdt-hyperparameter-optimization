from __future__ import division
import numpy as np
import os
import gzip
import urllib
import shutil
from tthAnalysis.bdtHyperparameterOptimization import gridsearch_main as gm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf
from tthAnalysis.bdtHyperparameterOptimization import universal
dir_path = os.path.dirname(os.path.realpath(__file__))
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
os.makedirs(sample_dir)
for file in file_list:
    file_loc = os.path.join(sample_dir, file)
    file_url = os.path.join(main_url, file + '.gz')
    urllib.urlretrieve(file_url, file_loc + '.gz')
    with gzip.open(file_loc + '.gz', 'rb') as f_in:
        with open(file_loc, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
data_dict = mf.create_datasets(sample_dir, 16)


def test_single_paramset():
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
    result = gm.single_paramset(parameters, iterations, grid_size)
    assert result == expected


def test_initialize_values():
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


def test_single_paramset():
    parameters = [
        {'p_name': 'foo', 'range_end': 10, 'range_start': 1},
        {'p_name': 'bar', 'range_end': 5, 'range_start': 1}
    ]
    iterations = [0, 1]
    grid_size = 2
    parameter_dict = gm.sindle_paramset(parameters, iterations, grid_size)
    expected = {'foo': 1, 'bar': 5}
    assert parameter_dict == expected


def test_perform_gridsearch():
    grid_settings = {'nthread': 2, 'grid_size': 2, 'num_classes': 10}
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    param_file = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters.json'
    )
    parameters = universal.read_parameters(param_file)
    result_dict = gm.perform_gridsearch(
        parameters, xt.ensemble_fitnesses, data_dict, grid_settings)
    assert result_dict != None


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
