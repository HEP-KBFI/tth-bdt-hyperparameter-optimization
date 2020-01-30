'''
Optimization of kappa of d_roc
Call with 'python'

Usage: slurm_tth_analysis.py
'''


import importlib
import numpy as np
import os
import shutil
from tthAnalysis.bdtTraining import tth_data_handler as ttHxt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)



def main():
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
    channel = global_settings['channel']
    bdtType = global_settings['bdtType']
    trainvar = global_settings['trainvar']
    fnFile = '_'.join(['fn', channel])
    importString = "".join(['tthAnalysis.bdtTraining.', fnFile])
    cf = __import__(importString, fromlist=[''])
    nthread = global_settings['nthread']
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    universal.save_run_settings(output_dir)
    data, trainVars = ttHxt.tth_analysis_main(
        channel, bdtType, nthread,
        output_dir, trainvar, cf
    )
    data_dict = ttHxt.create_xgb_data_dict(data, trainVars, global_settings)
    print("::::::: Reading parameters :::::::")
    param_file = os.path.join(
        main_dir,
        'data',
        'xgb_parameters.json'
    )
    value_dicts = universal.read_parameters(param_file)
    settings_dir = os.path.join(output_dir, 'run_settings')
    pso_settings = pm.read_weights(settings_dir)
    parameter_dicts = xt.prepare_run_params(
        value_dicts, pso_settings['sample_size'])
    print("\n============ Starting hyperparameter optimization ==========\n")
    global_settings_path = os.path.join(
        main_dir, 'data', 'global_settings.json')
    new_global_settings_path = os.path.join(
        main_dir, 'data', 'd_roc_settings', 'global_settings')
    os.rename(global_settings_path, global_settings_path + '_')
    for i in range(5):
        new_settings = new_global_settings_path +  str(i) + '.json'
        shutil.copy(new_settings, global_settings_path)
        global_settings = universal.read_settings(settings_dir, 'global')
        output_dir = os.path.expandvars(global_settings['output_dir'])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        result_dict = pm.run_pso(
            data_dict, value_dicts, sm.run_iteration, parameter_dicts,
            output_dir
        )
        print("\n============ Saving results ================\n")
        universal.save_results(result_dict, output_dir, plot_extras=True)
        print("Results saved to " + str(output_dir))
    os.rename(global_settings_path + '_', global_settings_path)


if __name__ == '__main__':
    main()