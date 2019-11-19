from tthAnalysis.bdtHyperparameterOptimization import ga_selection as gs

POPULATION = [1, 2, 3]
FITNESSES = [0.4, 0.6, 0.8]
PROBABILITIES = [0.2, 0.3, 0.5]


def test_selection():
    result = gs.tournament(POPULATION, FITNESSES)
    assert len(result) == 2, 'test_tournament failed'
    for member in result:
        assert member in POPULATION, 'test_tournament failed'


def test_roulette():
    result = gs.roulette(POPULATION, FITNESSES)
    assert len(result) == 2, 'test_roulette failed'
    for member in result:
        assert member in POPULATION, 'test_roulette failed'


def test_rank():
    result = gs.rank(POPULATION, FITNESSES)
    assert len(result) == 2, 'test_rank failed'
    for member in result:
        assert member in POPULATION, 'test_rank failed'


def test_normalize():
    result = gs.normalize(FITNESSES)
    assert sum(result) == 1, 'test_normalize failed'


def test_wheel():
    result = gs.wheel(POPULATION, PROBABILITIES)
    assert len(result) == 2, 'test_wheel failed'
    for member in result:
        assert member in POPULATION, 'test_wheel failed'
