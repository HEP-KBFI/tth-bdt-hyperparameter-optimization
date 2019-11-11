'''
Call with 'python3'

Usage: slurm_fitness.py --parameterFile=PTH --sample_dir=DIR --nthread=INT

Options:
    -p --parameterFile=PTH      Path to parameters to be run
    -s --sample_dir=DIR         Directory of the sample
    -n --nthread=INT            Number of threads to use

'''
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization.universal import read_parameters
from tthAnalysis.bdtHyperparameterOptimization.mnist_filereader import create_datasets
from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import parameter_evaluation
import docopt
import json
from pathlib import Path
import os
import csv
from __future__ import division


def save_info(score, pred_train, pred_test, saveDir):
    train_path = os.path.join(saveDir, "pred_train.lst")
    test_path = os.path.join(saveDir, "pred_test.lst")
    score_path = os.path.join(saveDir, "score.txt")
    with open(train_path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(pred_train)
    with open(test_path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(pred_test)
    with open(score_path, "w") as f:
        f.write(str(score))


def main(parameterFile, sample_dir, nthread):
    parameter_dict = read_parameters(parameterFile)[0]
    data_dict = create_datasets(sample_dir, nthread)
    path = Path(parameterFile)
    saveDir = str(path.parent)
    score, pred_train, pred_test = parameter_evaluation(
        parameter_dict, data_dict, nthread)
    save_info(score, pred_train, pred_test, saveDir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameterFile = arguments['--parameterFile']
        nthread = int(arguments['--nthread'])
        sample_dir = arguments['--sample_dir']
        main(parameterFile, sample_dir, nthread)
    except docopt.DocoptExit as e:
        print(e)