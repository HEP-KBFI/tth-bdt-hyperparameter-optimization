from __future__ import division
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
import os 
import shutil
import urllib
import gzip
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf
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
    result = xt.initialize_values(value_dicts)
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
    result = xt.prepare_run_params(
        value_dicts,
        sample_size
    )
    sum = 0
    for i in result:
        if isinstance(i['test1'], int):
            sum +=1
    assert len(result) == 3
    assert sum == 3


def test_parameter_evaluation():
    parameter_dict = {
            'num_boost_round': 71,
            'learning_rate': 0.07,
            'max_depth': 2,
            'gamma': 1.9,
            'min_child_weight': 18,
            'subsample': 0.9,
            'colsample_bytree': 0.8
    }
    nthread = 8
    num_class = 10
    results = xt.parameter_evaluation(
        parameter_dict, data_dict, nthread, num_class)
    assert results != None


def test_ensemble_fitnesses():
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
    global_settings = {'num_classes': 10, 'nthread': 8}
    results = xt. ensemble_fitnesses(
        parameter_dicts, data_dict, global_settings)
    assert results != None


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
