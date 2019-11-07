from tthAnalysis.bdtHyperparameterOptimization import plot_methods as pm
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(dir_path, "resources")


def test_read_parameters():
    path = os.path.join(resources_dir, "best_parameters.json")
    result = pm.read_parameters(path)
    assert len(result) == 3
