'''
Plot different method results to one plot (ROC)
Call with 'python3'

Usage: plot_methods.py --sampleDir=DIR --nthread=INT --parentDir=DIR --outputDir=DIR

Options:
    --sampleDir=DIR         Directory of the sample
    --nthread=INT           Number of threads to use
    --parentDir=DIR         Path to the parameters file
    --outputDir=DIR         Directory for plots and parameters

'''
import matplotlib.pyplot as plt
import xgboost
import glob
import os
from tthAnalysis.bdtHyperparameterOptimization import global_functions
from tthAnalysis.bdtHyperparameterOptimization import roccurve as rc
import json
import docopt


def parse_directories(parentDir):
    paths = []
    wild_card_path = os.path.join(parentDir, '*', '*.json')
    for path in glob.glob(wild_card_path):
        paths.append(path)
    return paths


def plot_single(data_dict, pred_test, pred_train, path):
    foldername = path.split("/")[-2]
    x_train, y_train = rc.roc(
        data_dict['training_labels'], pred_train)
    x_test, y_test = rc.roc(
        data_dict['testing_labels'], pred_test)
    plt.plot(
        x_train, y_train, lw=0.75, linestyle='--',
        label='%s train'%(foldername), zorder=100
    )
    plt.plot(
        x_test, y_test, lw=0.75, linestyle='-',
        label='%s test'%(foldername)
    )


def read_parameters(path):
    value_dicts = global_functions.read_parameters(path)
    parameters = value_dicts[0]
    return parameters
#### add default parameters


def main(parentDir, nthread, sampleDir, outputDir):
    data_dict = global_functions.create_datasets(sampleDir, nthread)
    paths = parse_directories(parentDir)
    for i, path in enumerate(paths):
        result_dict = {}
        parameters = read_parameters(path)
        parameters['nthread'] = nthread
        score, pred_train, pred_test = global_functions.parameter_evaluation(
            parameters, data_dict)
        result_dict['pred_train'] = pred_train
        result_dict['pred_test'] = pred_test
        result_dict['data_dict'] = data_dict
        result_dict['best_parameters'] = parameters
        outputDir1 = os.path.join(outputDir, str(i))
        if not os.path.exists(outputDir1):
            os.makedirs(outputDir1)
        assessment = global_functions.main_f1_calculate(
            result_dict['pred_train'],
            result_dict['pred_test'],
            result_dict['data_dict']
        )
        assessment['train_AUC'] = (-1) * train_AUC
        assessment['test_AUC'] = (-1) * test_AUC
        global_functions.best_to_file(
            parameters, outputDir1, assessment)
        plot_single(data_dict, pred_test, pred_train, path)
    plotOut = os.path.join(outputDir, 'roc_combined.png')
    plt.xlabel('Proportion of false values')
    plt.ylabel('Proportion of true values')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlim(0.0,0.1)
    plt.ylim(0.9,1.0)
    plt.tick_params(top=True, right=True, direction='in')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(plotOut)
    plt.close('all')


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parentDir = arguments['--parentDir']
        nthread = int(arguments['--nthread'])
        sampleDir = arguments['--sampleDir']
        outputDir = arguments['--outputDir']
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)
    except docopt.DocoptExit as e:
        print(e)
    main(parentDir, nthread, sampleDir, outputDir)
