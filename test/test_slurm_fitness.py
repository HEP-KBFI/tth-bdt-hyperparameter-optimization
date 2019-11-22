from __future__ import division
from tthAnalysis.bdtHyperparameterOptimization import slurm_fitness  as sm
import numpy as np
import os 
import shutil
import glob
import gzip
import shutil
import urllib
dir_path = os.path.dirname(os.path.realpath(__file__))
resourcesDir = os.path.join(dir_path, 'resources')
tmp_folder = os.path.join(resourcesDir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)


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
        urllib.urlretrieve(fileUrl, fileLoc + '.gz')
        with gzip.open(fileLoc + '.gz', 'rb') as f_in:
            with open(fileLoc, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    sm.main(parameterFile_newLoc, sampleDir, nthread)


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)