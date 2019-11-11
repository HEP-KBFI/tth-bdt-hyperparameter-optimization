from __future__ import division
from tthAnalysis.bdtHyperparameterOptimization.slurm_fitness import main
from tthAnalysis.bdtHyperparameterOptimization.slurm_fitness import save_info
import numpy as np
import os 
import shutil
import glob
import gzip
import shutil
import urllib2
dir_path = os.path.dirname(os.path.realpath(__file__))
resourcesDir = os.path.join(dir_path, 'resources')
tmp_folder = os.path.join(resourcesDir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)


def test_save_info():
    pred_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    pred_test = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    score = 2
    saveDir = tmp_folder
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    save_info(score, pred_train, pred_test, saveDir)
    train_path = os.path.join(saveDir, 'pred_train.lst')
    test_path = os.path.join(saveDir, 'pred_test.lst')
    score_path = os.path.join(saveDir, 'score.txt')
    count1 = len(open(train_path).readlines())
    count2 = len(open(test_path).readlines())
    count3 = len(open(score_path).readlines())
    assert count1 == count2 and count2 == 3
    assert count3 == 1


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


def test_main():
    main_url = 'http://yann.lecun.com/exdb/mnist/'
    train_images = 'train-images-idx3-ubyte'
    train_labels = 'train-labels-idx1-ubyte'
    test_images = 't10k-images-idx3-ubyte'
    test_labels = 't10k-labels-idx1-ubyte'
    file_list = [train_labels, train_images, test_labels, test_images]
    sampleDir = os.path.join(tmp_folder, 'samples_mnist')
    nthread = 2
    os.makedirs(sampleDir)
    parameterFile = os.path.join(resourcesDir, "xgbParameters.json")
    parameterFile_newLoc = os.path.join(
        tmp_folder, 'xgbParameters.json')
    shutil.copy(parameterFile, parameterFile_newLoc)
    for file in file_list:
        fileLoc = os.path.join(sampleDir, file)
        fileUrl = os.path.join(main_url, file + '.gz')
        urllib2.urlopen(fileUrl, fileLoc + '.gz')
        with gzip.open(fileLoc + '.gz', 'rb') as f_in:
            with open(fileLoc, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    main(parameterFile_newLoc, sampleDir, nthread)


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)