from __future__ import division
from subprocess import call
import os
import shutil
import gzip
import shutil
import urllib
import pytest
dir_path = os.path.dirname(os.path.realpath(__file__))
resourcesDir = os.path.join(dir_path, 'resources')
tmp_folder = os.path.join(resourcesDir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

main_url = 'http://yann.lecun.com/exdb/mnist/'
train_images = 'train-images-idx3-ubyte'
train_labels = 'train-labels-idx1-ubyte'
test_images = 't10k-images-idx3-ubyte'
test_labels = 't10k-labels-idx1-ubyte'
file_list = [train_labels, train_images, test_labels, test_images]
sampleDir = os.path.join(tmp_folder, 'samples_mnist')
nthread = 2
if not os.path.exists(sampleDir):
    os.makedirs(sampleDir)
parameterFile = os.path.join(resourcesDir, "xgbParameters.json")
parameterFile_newLoc = os.path.join(
    tmp_folder, 'xgbParameters.json')
shutil.copy(parameterFile, parameterFile_newLoc)
for file in file_list:
    fileLoc = os.path.join(sampleDir, file)
    fileUrl = os.path.join(main_url, file + '.gz')
    urllib.urlretrieve(fileUrl, fileLoc + '.gz')
    with gzip.open(fileLoc + '.gz', 'rb') as f_in:
        with open(fileLoc, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
cmssw_base = os.path.expandvars('$CMSSW_BASE')
main_dir = os.path.join(
    cmssw_base,
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
param_file = os.path.join(
    main_dir, 'test', 'resources', 'parameter_0.sh')


def test_slurm_pso_mnist():
    script = os.path.join(
        cmssw_base,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'scripts',
        'slurm_pso_mnist.py')
    fail = False
    try:
        call("python " + str(script), shell=True)
    except:
        fail = True
    assert not fail


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


def test_quasar_pso_mnist():
    script = os.path.join(
        cmssw_base,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'scripts',
        'quasar_pso_mnist.py')
    fail = False
    try:
        call("python " + str(script), shell=True)
    except:
        fail = True
    assert not fail


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


def test_ga_quasar_mnist():
    script = os.path.join(
        cmssw_base,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'scripts',
        'ga_quasar_mnist.py')
    fail = False
    try:
        call("python " + str(script), shell=True)
    except:
        fail = True
    assert not fail


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


def test_slurm_fitness_tth():
    script = os.path.join(
        cmssw_base,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'scripts',
        'slurm_fitness_tth.py')
    fail = False
    try:
        call("python " + str(script) + '-p' + str(param_file), shell=True)
    except:
        fail = True
    assert not fail


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


def test_slurm_fitness_mnist():
    script = os.path.join(
        cmssw_base,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'scripts',
        'slurm_fitness_mnist.py')
    fail = False
    try:
        call("python " + str(script) + '-p' + str(param_file), shell=True)
    except:
        fail = True
    assert not fail


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


os.rename(global_settings_path + '_', global_settings_path)
os.rename(pso_settings_path + '_', pso_settings_path)
