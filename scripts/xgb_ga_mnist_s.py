'''
Genetic algorithm for the hyperparameters optimization of XGBoost.
(MNIST numbers). Version for slurm.
Call with 'python'

Usage: slurm_ga_mnist.py
'''
from __future__ import division
import os
import warnings
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader  as mf
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_main as ga
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
warnings.filterwarnings('ignore', category=DeprecationWarning)


def main():
    print('::::::: Reading GA settings & XGBoost parameters :::::::')
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
    settings_dir = os.path.join(
        main_dir, 'data')
    global_settings = universal.read_settings(settings_dir, 'global')

    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    settings_dict = universal.read_settings('ga')
    settings_dict.update(global_settings)

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
        sm.run_iteration
    )
    universal.save_results(result, output_dir)
    sm.clear_from_files(global_settings)


if __name__ == '__main__':
    main()
