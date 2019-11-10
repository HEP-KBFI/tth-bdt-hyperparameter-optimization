from tthAnalysis.bdtHyperparameterOptimization import roccurve as rc
import numpy as np
import os 
import shutil
dir_path = os.path.dirname(os.path.realpath(__file__))
resourcesDir = os.path.join(dir_path, "resources", "tmp")


def test_create_pairs():
    matrix = [[3, 7, 7], [4, 5, 6], [2, 9, 8]]
    expected = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]
    result = rc.create_pairs(matrix)
    assert result == expected


def test_normalization():
    elems = (4, 6)
    expected = (0.4, 0.6)
    result = rc.normalization(elems)
    assert result == expected


def test_create_mask():
    true_labels = [1, 0, 5, 4, 2]
    pair = (1, 3)
    expected = [True, False, False, False, False]
    result = rc.create_mask(true_labels, pair)
    assert result == expected


def test_get_values():
    matrix = [[3, 7, 7], [4, 5, 6], [2, 9, 8]]
    pair = (0, 2)
    expected = [
        (0.3, 0.7),
        (0.4, 0.6),
        (0.2, 0.8)
    ]
    result = rc.get_values(matrix, pair)
    assert result == expected


def test_choose_values():
    matrix = [[3, 7, 7], [4, 5, 6], [2, 9, 8]]
    pair = (0, 2)
    true_labels = [1, 2, 2]
    labelsOut = np.array([2, 2])
    elemsOut = np.array([
        [0.4, 0.6],
        [0.2, 0.8]
    ])
    result = rc.choose_values(matrix, pair, true_labels)
    print(result[1])
    assert (result[0] == labelsOut).all()
    assert (result[1] == elemsOut).all()


def test_plot_costFunction():
    avg_scores = [0.9, 0.95, 0.99, 1]
    error = False
    try:
        rc.plot_costFunction(avg_scores, resourcesDir)
    except:
        error = True
    assert error == False


def test_dummy_delete_files():
    if os.path.exists(resourcesDir):
        shutil.rmtree(resourcesDir)