'''
Call with 'python3'

Usage: slurm_fitness.py --parameter_file=PTH --output_dir=DIR

Options:
    -p --parameter_file=PTH      Path to parameters to be run
    --output_dir=DIR             Directory of the output


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


def main(parameter_file, output_dir):
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = universal.read_settings(settings_dir, 'global')
    num_classes = global_settings['num_classes']
    sample_dir = global_settings['sample_dir']
    nthread = global_settings['nthread']
    parameter_dict = universal.read_parameters(parameter_file)[0]
    data_dict = mf.create_datasets(sample_dir, nthread)
    path = Path(parameter_file)
    saveDir = str(path.parent)
    score, pred_train, pred_test, feature_importance = xt.parameter_evaluation(
        parameter_dict, data_dict, nthread, num_classes)
    sm.save_info(score, pred_train, pred_test, saveDir, feature_importance)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        output_dir = arguments['--output_dir']
        main(parameter_file, output_dir)
    except docopt.DocoptExit as e:
        print(e)