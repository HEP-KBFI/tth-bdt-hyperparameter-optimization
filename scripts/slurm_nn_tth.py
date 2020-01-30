'''
Call with 'python'

Usage: slurm_nn_tth.py --parameter_file=PTH --output_dir=DIR

Options:
    -p --parameter_file=PTH      Path to parameters to be run
    --output_dir=DIR             Directory of the output

'''
from __future__ import division
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import nn_tools as nnt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtTraining import trainvar_choice as tc
from tthAnalysis.bdtTraining import tth_data_handler as ttHxt
import docopt
import json
from pathlib import Path
import os
import csv


def main(parameter_file, output_dir):
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = universal.read_settings(settings_dir, 'global')
    num_classes = global_settings['num_classes']
    channel = global_settings['channel']
    bdtType = global_settings['bdtType']
    trainvar = global_settings['trainvar']
    fnFile = '_'.join(['fn', channel])
    importString = "".join(['tthAnalysis.bdtTraining.', fnFile])
    if bool(int(global_settings['trainvar_opt'])):
        cf = tc
    else:
        cf = __import__(importString, fromlist=[''])
    nthread = global_settings['nthread']
    sample_dir = global_settings['sample_dir']
    data, trainVars = ttHxt.tth_analysis_main(
        channel, bdtType, nthread,
        output_dir, trainvar, cf
    )
    data_dict = ttHxt.create_nn_data_dict(
        data, trainVars, global_settings)
    parameter_dict = universal.read_parameters(parameter_file)[0]
    path = Path(parameter_file)
    save_dir = str(path.parent)
    score, pred_train, pred_test, feature_importance = nnt.parameter_evaluation(
        parameter_dict, data_dict, nthread, num_classes)
    universal.save_info(
        score, pred_train, pred_test, save_dir, feature_importance)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        output_dir = arguments['--output_dir']
        main(parameter_file, output_dir)
    except docopt.DocoptExit as e:
        print(e)