'''
Call with 'python3'

Usage: slurm_fitness.py --parameterFile=PTH --sample_dir=DIR --nthread=INT

Options:
    -p --parameterFile=PTH      Path to parameters to be run
    -s --sample_dir=DIR         Directory of the sample
    -n --nthread=INT            Number of threads to use

'''
from __future__ import division
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
import docopt
import json
from pathlib import Path
import os
import csv


def main(parameterFile, sample_dir, nthread):
    parameter_dict = universal.read_parameters(parameterFile)[0]
    data_dict = mf.create_datasets(sample_dir, nthread)
    path = Path(parameterFile)
    saveDir = str(path.parent)
    score, pred_train, pred_test = xt.parameter_evaluation(
        parameter_dict, data_dict, nthread)
    sm.save_info(score, pred_train, pred_test, saveDir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameterFile = arguments['--parameterFile']
        nthread = int(arguments['--nthread'])
        sample_dir = arguments['--sample_dir']
        main(parameterFile, sample_dir, nthread)
    except docopt.DocoptExit as e:
        print(e)