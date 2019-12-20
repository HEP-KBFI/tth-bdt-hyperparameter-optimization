from __future__ import division
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
import numpy as np
import os 
import shutil
import urllib
import gzip
import glob
import timeit
import timeout_decorator
import subprocess
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf
dir_path = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(dir_path, 'resources')
tmp_folder = os.path.join(resources_dir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)


def test_parameters_to_file():
    output_dir = os.path.join(tmp_folder, 'slurm')
    parameter_dict1 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dict2 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dict3 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dicts = [
        parameter_dict1,
        parameter_dict2,
        parameter_dict3
    ]
    sm.parameters_to_file(output_dir, parameter_dicts)
    wild_card_path = os.path.join(output_dir, '*', '*')
    number_files = len(glob.glob(wild_card_path))
    assert number_files == 3


def test_lists_from_file():
    path = os.path.join(resources_dir, 'samples', '2', 'pred_test.lst')
    result = sm.lists_from_file(path)
    expected = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
    assert result == expected


def test_create_result_lists():
    output_dir = os.path.join(resources_dir)
    result = sm.create_result_lists(output_dir, 'pred_test')
    expected = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ], dtype=int)
    assert (result == expected).all()


def test_get_sample_nr():
    path = "/foo/1/bar.baz"
    expected = 1
    result = sm.get_sample_nr(path)
    assert result == expected


@timeout_decorator.timeout(10)
def test_wait_iteration():
    working = False
    try:
        sm.wait_iteration(resources_dir, 2)
    except SystemExit: 
        working = True
    assert working


def test_delete_previous_files():
    samplespath = os.path.join(resources_dir, 'samples')
    dest_folder = os.path.join(tmp_folder, 'samples')
    subprocess.call(['cp', '-r', samplespath, dest_folder])
    sm.delete_previous_files(tmp_folder)
    wild_card_path1 = os.path.join(dest_folder, '*', '*.txt')
    wild_card_path2 = os.path.join(dest_folder, '*', '*.lst')
    number_files1 = len(glob.glob(wild_card_path1))
    number_files2 = len(glob.glob(wild_card_path2))
    number_files = number_files1 + number_files2
    assert number_files == 0


def test_read_fitness():
    result = sm.read_fitness(resources_dir)[0]
    expected = {"foo": 1, "bar": 2, "baz": 3}
    assert result == expected


def test_check_error():
    error = False
    try:
        sm.check_error(resources_dir)
    except:
        error = True
    assert error


def test_prepare_job_file():
    global_settings = {
        'output_dir': tmp_folder,
        'sample_type': 'mnist',
        'nthread': 2
    }
    nthread = 2
    parameterFile = os.path.join(resources_dir, 'xgbParameters.json')
    sample_dir = os.path.join(resources_dir, 'samples')
    job_nr = 1
    output_dir = os.path.join(resources_dir, 'tmp')
    templateDir = resources_dir
    sm.prepare_job_file(
        parameterFile, job_nr, global_settings)
    jobFile = os.path.join(resources_dir, 'tmp', 'parameter_1.sh')
    with open(jobFile, 'r') as f:
        number_lines = len(f.readlines())
    assert number_lines == 10


def test_dummy_delete_files():
    if os.path.exists(os.path.join(tmp_folder)):
        shutil.rmtree(os.path.join(tmp_folder))


def test_save_info():
    pred_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    pred_test = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    score = {'f1_score': 0.8, 'g_score': 0.79}
    feature_importances = {'f0': 12, 'f1': 0.5, 'f2': 50.2}
    save_dir = tmp_folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sm.save_info(score, pred_train, pred_test, save_dir, feature_importances)
    train_path = os.path.join(save_dir, 'pred_train.lst')
    test_path = os.path.join(save_dir, 'pred_test.lst')
    score_path = os.path.join(save_dir, 'score.json')
    count1 = len(open(train_path).readlines())
    count2 = len(open(test_path).readlines())
    count3 = len(open(score_path).readlines())
    assert count1 == count2 and count2 == 3
    assert count3 == 1


def test_clear_from_files():
    settings = {'output_dir': tmp_folder}
    for i in range(5):
        path = os.path.join(tmp_folder, 'parameter_' + str(i) +'.sh')
        open(path, 'w').close()
    sample_dir = os.path.join(tmp_folder, 'samples')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    sm.clear_from_files(settings)
    wild_card_path = os.path.join(tmp_folder, 'parameter_*.sh')
    assert len(glob.glob(wild_card_path)) == 0
    assert not os.path.isdir(sample_dir)


def test_read_feature_importances():
    expected = [
        {'f1': 1, 'f2': 2, 'f3': 3},
        {'f1': 1, 'f2': 2, 'f3': 3},
        {'f1': 1, 'f2': 2, 'f3': 3}
    ]
    result = sm.read_feature_importances(resources_dir)
    assert result == expected


def test_run_iteration():
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
    global_settings = {
        'sample_dir': sample_dir,
        'nthread': 8,
        'sample_type': 'mnist',
        'output_dir': tmp_folder,
        'optimization_algo': 'pso',
        'num_classes': 10}
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
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
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
    score_dicts, pred_trains, pred_tests, feature_importances = sm.run_iteration(
        parameter_dicts, data_dict, global_settings)
    os.rename(global_settings_path + '_', global_settings_path)
    os.rename(pso_settings_path + '_', pso_settings_path)
    assert score_dicts != None


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
