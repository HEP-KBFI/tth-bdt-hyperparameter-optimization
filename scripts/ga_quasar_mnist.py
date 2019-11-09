"""
Genetic algorithm for optimizing the hyperparameters of XGBoost. (MNIST nbr)
Call with 'python3'

Usage: mnist_ga.py --sample_dir=DIR --nthread=INT --output_dir=DIR --param_file=PTH --sett_file=PTH

Options:
    --sample_dir=DIR        The location of MNIST number sample
    --nthread=INT           Number of threads to be used (recommended = 8 cores)
    --output_dir=DIR        Output directory for the results
    --param_file=PTH        Path to parameters file
    --sett_file=PTH         Path to settings file

"""
import os
import docopt
from tthAnalysis.bdtHyperparameterOptimization.universal import read_parameters
from tthAnalysis.bdtHyperparameterOptimization.universal import save_results
from tthAnalysis.bdtHyperparameterOptimization.mnist_filereader import create_datasets
from tthAnalysis.bdtHyperparameterOptimization.ga_main import evolution


def main(sample_dir, nthread, output_dir, param_file, sett_file):

    # Load settings for genetic algorithm
    settings_dict = read_parameters(sett_file)[0]
    settings_dict.update({'nthread': nthread})

    # Load data
    data_dict = create_datasets(sample_dir, nthread)

    # Load parameters for optimization
    param_dict = read_parameters(param_file)

    # Run genetic algorithm and save results
    result = evolution(settings_dict, data_dict, param_dict, nthread)
    save_results(result, output_dir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        sample_dir = arguments['--sample_dir']
        nthread = int(arguments['--nthread'])
        output_dir = arguments['--output_dir']
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        param_file = arguments['--param_file']
        sett_file = arguments['--sett_file']
        main(sample_dir, nthread, output_dir, param_file, sett_file)
    except docopt.DocoptExit as e:
        print(e)