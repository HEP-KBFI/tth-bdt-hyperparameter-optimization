'''
Particle swarm optimization for the hyperparameters optimization of XGBoost.
(MNIST numbers). Version for quasar.
Call with 'python'

Usage: quasar_pso_mnist.py

'''
from __future__ import division
import numpy as np
import os
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader  as mf
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
np.random.seed(1)


def main():
    global_settings = universal.read_settings('global')
    output_dir = global_settings['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print("::::::: Loading data ::::::::")
    data_dict = create_datasets(
        global_settings['sample_dir'],
        global_settings['nthread'])
    print("::::::: Reading parameters :::::::")
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    param_file = os.path.join(
        cmssw_base_path,
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters'
    )
    value_dicts = universal.read_parameters(param_file)
    pso_settings = pm.read_weights(value_dicts)
    parameter_dicts = xt.prepare_run_params(
        global_settings['nthread'], value_dicts, pso_settings['sample_size'])
    result_dict = pm.run_pso(
        global_settings, pso_settings, data_dict,
        value_dicts, xt.ensemble_fitnesses, parameter_dicts
    )
    universal.save_results(result_dict, output_dir)


if __name__ == '__main__':
    main()

# import what scoring to use here ??