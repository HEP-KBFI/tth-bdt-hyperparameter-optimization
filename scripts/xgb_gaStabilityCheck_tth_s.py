'''
Stability check for the genetic algorithm.
Call with 'python'

Usage: xgb_gaStabilityCheck_tth_s.py
'''
from __future__ import division
import os
import numpy as np
from tthAnalysis.bdtTraining import xgb_tth as ttHxt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_main as ga
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtHyperparameterOptimization import stability_check_tools as sct

NUMBER_REPETITIONS = 10

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
    nthread = global_settings['nthread']
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    settings_dict = universal.read_settings(settings_dir, 'ga')
    settings_dict.update(global_settings)
    param_file = os.path.join(
        settings_dir, 'xgb_parameters.json')
    param_dict = universal.read_parameters(param_file)
    universal.save_run_settings(output_dir)

    print('::::::: Loading data ::::::::')
    channel = global_settings['channel']
    bdt_type = global_settings['bdtType']
    trainvar = global_settings['trainvar']
    fn_file = '_'.join(['fn', channel])
    import_string = "".join(['tthAnalysis.bdtTraining.', fn_file])
    cf = __import__(import_string, fromlist=[''])
    data, trainvars = ttHxt.tth_analysis_main(
        channel, bdt_type, nthread, output_dir, trainvar, cf)
    data = ttHxt.convert_data_to_correct_format(data)
    data_dict = ttHxt.createDataSet(data, trainvars, nthread)

    result_dicts = []
    for i in range(NUMBER_REPETITIONS):
        output_dir_single = os.path.join(output_dir, 'iteration_' + str(i))
        np.random.seed(i)
        result = ga.evolution(
            settings_dict,
            param_dict,
            data_dict,
            xt.prepare_run_params,
            sm.run_iteration
        )
        universal.save_results(result, output_dir_single)
        print(
            'Results of stability iteration '
            + str(i) + ' are saved to ' + str(output_dir)
        )
        result_dicts.append(result)

    best_parameter_dicts = [
        result_dict['best_parameters'] for result_dict in result_dicts
    ]
    keys = best_parameter_dicts[0].keys()
    dict_of_parameter_lists = universal.values_to_list_dict(
        keys, best_parameter_dicts)
    sct.stability_check_main(dict_of_parameter_lists, output_dir)
    sct.plot_all_radar_charts(dict_of_parameter_lists, output_dir)
    sct.plot_score_stability(result_dicts, output_dir)    


if __name__ == '__main__':
    main()
