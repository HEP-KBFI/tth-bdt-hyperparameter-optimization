'''
Grid search for best parameters to be compared with evol. algorithms
Call with 'python'

Usage: xgb_gridSearch.py
'''

from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import gridsearch_main as gsm
import docopt
import os


def main():
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    grid_settings = universal.read_settings('global')
    grid_settings.update({'grid_size': 2})
    param_file = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters.json'
    )
    parameters = universal.read_parameters(param_file)
    data_dict = mf.create_datasets(
        global_settings['sample_dir'], global_settings['nthread'])
    result_dict = gsm.perform_gridsearch(
        parameters,
        xt.ensemble_fitnesses,
        data_dict,
        grid_settings
    )
    universal.save_results(
        result_dict, global_settings['output_dir'], plot_roc=False)


if __name__ == '__main__':
    main()