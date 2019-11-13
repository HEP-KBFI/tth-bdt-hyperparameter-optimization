'''
Grid search for best parameters to be compared with evol. algorithms
Call with 'python3'

Usage: xgb_gridSearch.py --sample_dir=DIR --nthread=INT --param_file=PTH --outputDir=DIR

Options:
    --sample_dir=DIR        Directory of the sample
    --nthread=INT           Number of threads to use
    --param_file=PTH        Path to the parameters file
    --outputDir=DIR         Directory for plots and parameters

'''
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import gridSearch_main as gsm
import docopt
import os


GRID_SIZE = 2

def main(outputDir):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    parameters = universal.read_parameters(param_file)
    data_dict = universal.create_datasets(sample_dir, nthread)
    result_dict = gsm.perform_gridSearch(
        param_file,
        nthread,
        sample_dir,
        outputDir,
        pm.ensemble_fitness
    )
    universal.save_results(result_dict, outputDir, roc=False)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        param_file = arguments['--param_file']
        nthread = int(arguments['--nthread'])
        sample_dir = arguments['--sample_dir']
        outputDir = arguments['--outputDir']
        main(param_file, nthread, sample_dir, outputDir)
    except docopt.DocoptExit as e:
        print(e)