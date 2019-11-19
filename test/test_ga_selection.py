from tthAnalysis.bdtHyperparameterOptimization import ga_selection as gs

population = [1, 2, 3]
fitnesses = [0.4, 0.6, 0.8]
probabilities = [0.2, 0.3, 0.5]

def test_selection():
    result = gs.tournament(population, fitnesses)
    assert len(result) == 2, 'test_tournament failed'
    for member in result:
        assert member in population, 'test_tournament failed'


def test_roulette():
    result = gs.roulette(population, fitnesses)
    assert len(result) == 2, 'test_roulette failed'
    for member in result:
        assert member in population, 'test_roulette failed'


def test_rank():
    result = gs.rank(population, fitnesses)
    assert len(result) == 2, 'test_rank failed'
    for member in result:
        assert member in population, 'test_rank failed' 


def test_normalize():
    result = gs.normalize(fitnesses)
    assert sum(result) == 1, 'test_normalize failed'


def test_wheel():
    result = gs.wheel(population, probabilities)
    assert len(result) == 2, 'test_wheel failed'
    for member in result:
        assert member in population, 'test_wheel failed'
