from __future__ import division
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import parameters_to_file
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import lists_from_file
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import create_result_lists
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import get_sample_nr
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import wait_iteration
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import delete_previous_files
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import prepare_jobFile
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import check_error
from tthAnalysis.bdtHyperparameterOptimization.slurm_main import read_fitness
import numpy as np
import os 
import shutil
import glob
import timeit
import timeout_decorator
import subprocess
dir_path = os.path.dirname(os.path.realpath(__file__))
resourcesDir = os.path.join(dir_path, 'resources')
tmp_folder = os.path.join(resourcesDir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)


def test_parameters_to_file():
    outputDir = os.path.join(tmp_folder, 'slurm')
    parameter_dict1 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dict2 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dict3 = {'a': 1, 'b': 2, 'c': 3}
    parameter_dicts = [
        parameter_dict1,
        parameter_dict2,
        parameter_dict3
    ]
    parameters_to_file(outputDir, parameter_dicts)
    wild_card_path = os.path.join(outputDir, '*', '*')
    number_files = len(glob.glob(wild_card_path))
    assert number_files == 3


def test_lists_from_file():
    path = os.path.join(resourcesDir, 'samples', '2', 'pred_test.lst')
    result = lists_from_file(path)
    expected = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
    assert result == expected


def test_create_result_lists():
    outputDir = os.path.join(resourcesDir)
    result = create_result_lists(outputDir, 'pred_test')
    expected = np.array([
        [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ], dtype=int)
    assert (result == expected).all()


def test_get_sample_nr():
    path = "/foo/1/bar.baz"
    expected = 1
    result = get_sample_nr(path)
    assert result == expected


@timeout_decorator.timeout(10)
def test_wait_iteration():
    start = timeit.timeit()
    wait_iteration(resourcesDir, 2)
    end = timeit.timeit()
    assert end - start < 1


def test_delete_previous_files():
    samplesmath = os.path.join(resourcesDir, 'samples')
    destFolder = os.path.join(tmp_folder, 'samples')
    subprocess.call(['cp', '-r', samplesmath, destFolder])
    delete_previous_files(tmp_folder)
    wild_card_path1 = os.path.join(destFolder, '*', '*.txt')
    wild_card_path2 = os.path.join(destFolder, '*', '*.lst')
    number_files1 = len(glob.glob(wild_card_path1))
    number_files2 = len(glob.glob(wild_card_path2))
    number_files = number_files1 + number_files2
    assert number_files == 0


def test_read_fitness():
    result = read_fitness(resourcesDir)[0]
    expected = 1
    assert result == expected


def test_check_error():
    error = False
    try:
        check_error(resourcesDir)
    except:
        error = True
    assert error == True


def test_prepare_jobFile():
    nthread = 2
    parameterFile = os.path.join(resourcesDir, 'xgbParameters.json')
    sample_dir = os.path.join(resourcesDir, 'samples')
    job_nr = 1
    outputDir = os.path.join(resourcesDir, 'tmp')
    templateDir = resourcesDir
    prepare_jobFile(
        parameterFile, sample_dir, nthread, job_nr, outputDir, templateDir)
    jobFile = os.path.join(resourcesDir, 'tmp', 'parameter_1.sh')
    with open(jobFile, 'r') as f:
        number_lines = len(f.readlines())
    assert number_lines == 10


def test_dummy_delete_files():
    if os.path.exists(os.path.join(tmp_folder)):
        shutil.rmtree(os.path.join(tmp_folder))