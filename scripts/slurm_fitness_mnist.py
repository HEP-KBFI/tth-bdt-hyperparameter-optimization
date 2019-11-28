'''
Call with 'python3'

Usage: slurm_fitness.py --parameter_file=PTH

Options:
    -p --parameter_file=PTH      Path to parameters to be run


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
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)



def main(parameter_file):
    global_settings = universal.read_settings('global')
    num_classes = global_settings['num_classes']
    sample_dir = global_settings['sample_dir']
    nthread = global_settings['nthread']
    parameter_dict = universal.read_parameters(parameter_file)[0]
    data_dict = mf.create_datasets(sample_dir, nthread)
    path = Path(parameter_file)
    saveDir = str(path.parent)
    score, pred_train, pred_test = xt.parameter_evaluation(
        parameter_dict, data_dict, nthread, num_classes)
    sm.save_info(score, pred_train, pred_test, saveDir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        main(parameter_file)
    except docopt.DocoptExit as e:
        print(e)