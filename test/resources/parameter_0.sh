#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=500
#SBATCH --cpus-per-task=8
#SBATCH -e /home/laurits/CMSSW_10_2_10/src/tthAnalysis/bdtHyperparameterOptimization/test/resources/tmp/error
#SBATCH -o /home/laurits/CMSSW_10_2_10/src/tthAnalysis/bdtHyperparameterOptimization/test/resources/tmp/output
python /home/laurits/CMSSW_10_2_10/src/tthAnalysis/bdtHyperparameterOptimization/scripts/slurm_fitness_mnist.py --parameter_file /home/laurits/CMSSW_10_2_10/src/tthAnalysis/bdtHyperparameterOptimization/test/resources/samples/0/parameters.json
        