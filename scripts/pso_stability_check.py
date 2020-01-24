'''
Particle swarm optimization for the hyperparameters optimization of XGBoost.
(MNIST numbers). Version for quasar.
Call with 'python'

Usage: quasar_pso_mnist.py

'''
from __future__ import division
import numpy as np
import os
from tthAnalysis.bdtTraining import xgb_tth as ttHxt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import stability_check_tools as sct


NUMBER_REPETITIONS = 50

def main():
    print("::::::: Loading data ::::::::")
    global_settings = universal.read_settings('global')
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
    data, trainVars = ttHxt.tth_analysis_main(
        channel, bdtType, nthread,
        output_dir, trainvar, cf
    )
    data = ttHxt.convert_data_to_correct_format(data)
    data_dict = ttHxt.createDataSet(data, trainVars, nthread)
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
    pso_settings = pm.read_weights()
    result_dicts = []
    for i in range(NUMBER_REPETITIONS):
        output_dir_single = os.path.join(output_dir, 'iteration_' + str(i))
        np.random.seed(i)
        parameter_dicts = xt.prepare_run_params(
            value_dicts, pso_settings['sample_size'])
        result_dict = pm.run_pso(
            data_dict, value_dicts, sm.run_iteration, parameter_dicts
        )
        universal.save_results(result_dict, output_dir_single)
        print(
            'Results of stability iteration '
            + str(i) + ' are saved to ' + str(output_dir)
        )
        sm.clear_from_files(global_settings)
        result_dicts.append(result_dict)
    keys = result_dicts[0].keys()
    best_parameter_dicts = [
        result_dict["best_parameters"] for result_dict in result_dicts
    ]
    dict_of_parameter_lists = universal.values_to_list_dict(
        keys, best_parameter_dicts)
    sct.stability_check_main(dict_of_parameter_lists, output_dir)
    sct.plot_all_radar_charts(result_dict['best_parameters'], output_dir)
    sct.plot_score_stability(result_dicts, output_dir)


if __name__ == '__main__':
    main()