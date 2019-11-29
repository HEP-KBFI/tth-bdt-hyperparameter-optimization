'''
Genetic algorithm for optimizing the hyperparameters of XGBoost. (MNIST nbr)
Call with 'python'

Usage: ga_quasar_mnist.py --sample_dir=DIR --nthread=INT --output_dir=DIR --param_file=PTH --sett_file=PTH

Options:
    --sample_dir=DIR        The location of MNIST number sample
    --nthread=INT           Number of threads to be used (recommended = 8 cores)
    --output_dir=DIR        Output directory for the results
    --param_file=PTH        Path to parameters file

'''
from __future__ import division
import os
import docopt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf
from tthAnalysis.bdtHyperparameterOptimization import ga_main as ga


def main(sample_dir, nthread, output_dir, param_file, sett_file):

    print('::::::: Reading GA settings & XGBoost parameters :::::::')
    settings_dict = universal.read_settings('ga')
    settings_dict.update({'nthread': nthread})

    # Load parameters for optimization
    param_dict = universal.read_parameters(param_file)

    print('::::::: Loading data ::::::::')
    data_dict = mf.create_datasets(sample_dir, nthread)

    # Run genetic algorithm and save results
    result = ga.evolution(settings_dict, data_dict, param_dict)
    universal.save_results(result, output_dir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        sample_dir = arguments['--sample_dir']
        nthread = int(arguments['--nthread'])
        output_dir = arguments['--output_dir']
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        param_file = arguments['--param_file']
        main(sample_dir, nthread, output_dir, param_file, sett_file)
    except docopt.DocoptExit as e:
        print(e)
