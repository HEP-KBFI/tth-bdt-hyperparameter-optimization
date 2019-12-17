'''
Call with 'python'

Usage: slurm_fitness_tth.py --parameter_file=PTH

Options:
    -p --parameter_file=PTH      Path to parameters to be run

'''
from __future__ import division
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtTraining import xgb_tth as ttHxt
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
    channel = global_settings['channel']
    bdtType = global_settings['bdtType']
    trainvar = global_settings['trainvar']
    output_dir = os.path.expandvars(global_settings['output_dir'])
    fnFile = '_'.join(['fn', channel])
    importString = "".join(['tthAnalysis.bdtTraining.', fnFile])
    cf = __import__(importString, fromlist=[''])
    nthread = global_settings['nthread']
    sample_dir = global_settings['sample_dir']
    data, trainVars = ttHxt.tth_analysis_main(
        channel, bdtType, nthread,
        output_dir, trainvar, cf
    )
    data_dict = ttHxt.createDataSet(
        data, trainVars, nthread)
    parameter_dict = universal.read_parameters(parameter_file)[0]
    path = Path(parameter_file)
    saveDir = str(path.parent)
    score, pred_train, pred_test, feature_importance = xt.parameter_evaluation(
        parameter_dict, data_dict, nthread, num_classes)
    sm.save_info(score, pred_train, pred_test, saveDir, feature_importance)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        main(parameter_file)
    except docopt.DocoptExit as e:
        print(e)