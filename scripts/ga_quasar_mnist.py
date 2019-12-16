'''
Genetic algorithm for the hyperparameters optimization of XGBoost.
(MNIST numbers).
Call with 'python'

Usage: ga_quasar_mnist.py
'''
from __future__ import division
import os
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import ga_main as ga


def main():
    print('::::::: Reading GA settings & XGBoost parameters :::::::')
    global_settings = universal.read_settings('global')

    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    settings_dict = universal.read_settings('ga')
    settings_dict.update(global_settings)

    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    param_file = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters.json'
    )
    param_dict = universal.read_parameters(param_file)

    print('::::::: Loading data ::::::::')
    sample_dir = os.path.expandvars(global_settings['sample_dir'])
    data_dict = mf.create_datasets(
        sample_dir,
        global_settings['nthread'])

    result = ga.evolution(
        settings_dict,
        data_dict,
        param_dict,
        xt.prepare_run_params,
        xt.ensemble_fitnesses
    )
    universal.save_results(result, output_dir)


if __name__ == '__main__':
    main()
