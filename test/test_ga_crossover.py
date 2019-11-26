'''Testing the crossover functions for the genetic algorithm
Missing tests for the following functions:
    kpoint_crossover
    uniform_crossover
    group_crossover
    group_mutate
    mutation_fix
    encode_parent
    decode_offspring
    grouping
    degroup
'''
from tthAnalysis.bdtHyperparameterOptimization import ga_crossover as gc


def test_chromosome_mutate():
    '''Testing chromosome mutate function'''
    initial = '0101010101'

    # Mutation chance = 0
    result1 = gc.chromosome_mutate(initial, 0)
    assert initial == result1, 'test_chromosome_mutate failed'

    # Mutation chance = 1
    result2 = gc.chromosome_mutate(
        gc.chromosome_mutate(initial, 1), 1)
    assert initial == result2, 'test_chromosome_mutate failed'


def test_int_encoding():
    '''Testing integer encoding function'''
    initial = [2, 10, 500, 3000, 54000]
    expected = [
        '00000000000000000000000000000010',
        '00000000000000000000000000001010',
        '00000000000000000000000111110100',
        '00000000000000000000101110111000',
        '00000000000000001101001011110000'
    ]
    results = []
    for number in initial:
        results.append(gc.int_encoding(number))
    for i, result in enumerate(results):
        assert result == expected[i], 'test_int_encoding failed'


def test_float_encoding():
    '''Testing float encoding function'''
    initial = [0.1, 0.002, 0.00003, 0.0000004, 55.55]
    expected = [
        '00000000000000000010011100010000',
        '00000000000000000000000011001000',
        '00000000000000000000000000000011',
        '00000000000000000000000000000000',
        '00000000010101001100001100111000',
    ]
    results = []
    for number in initial:
        results.append(gc.float_encoding(number))
    for i, result in enumerate(results):
        assert result == expected[i], 'test_float_encoding failed'


def test_int_decoding():
    '''Testing integer decoding function'''
    initial = [
        '00000000000000000000000000000010',
        '00000000000000000000000000001010',
        '00000000000000000000000111110100',
        '00000000000000000000101110111000',
        '00000000000000001101001011110000'
    ]
    expected = [2, 10, 500, 3000, 54000]
    results = []
    for code in initial:
        results.append(gc.int_decoding(code))
    for i, result in enumerate(results):
        assert result == expected[i], 'test_int_decoding failed'


def test_float_decoding():
    '''Testing float decoding function'''
    initial = [
        '00000000000000000010011100010000',
        '00000000000000000000000011001000',
        '00000000000000000000000000000011',
        '00000000000000000000000000000000',
        '00000000010101001100001100111000'
    ]
    expected = [0.1, 0.002, 0.00003, 0.0, 55.55]
    results = []
    for code in initial:
        results.append(gc.float_decoding(code))
    for i, result in enumerate(results):
        assert result == expected[i], 'float_int_decoding failed'
