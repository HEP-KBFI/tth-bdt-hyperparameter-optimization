import numpy as np
import os
import warnings
import json
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtHyperparameterOptimization import atlas_tools as at
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_main as gm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt

np.random.seed(1)
path_to_file = "$HOME/training.csv"

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
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    universal.save_run_settings(output_dir)
    print("::::::: Loading data ::::::::")
    data_dict = at.create_atlas_data_dict(path_to_file, global_settings)
    print("::::::: Reading parameters :::::::")
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    param_file = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters.json'
    )
    value_dicts = universal.read_parameters(param_file)
    settings = universal.read_settings(settings_dir, 'ga')
    settings.update(global_settings)
    parameter_dicts = xt.prepare_run_params(
        value_dicts, settings['sample_size'])
    result_dict = gm.evolve(
        parameter_dicts, settings, value_dicts, data_dict,
        xt.prepare_run_params, sm.run_iteration
    )
    return result_dict, output_dir


if __name__ == '__main__':
    result_dict, output_dir = main()
    at.save_results(result_dict, output_dir)
    print("Results saved to " + str(output_dir))
    score_path = os.path.join(output_dir, 'score.json')
    score_dict = result_dict['score_dict']
    with open(score_path, 'w') as file:
        json.dump(score_dict, file)