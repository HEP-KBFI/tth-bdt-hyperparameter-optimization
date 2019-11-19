# Various selection methods that can be used by the genetic algorithm
from __future__ import division
import random
import numpy as np


# Tournament selection
def tournament(population, fitnesses, t_size=2, t_prob=0.8):

    # Initialization
    parents = []
    while len(parents) < 2:
        tournament = []
        t_fitness = []

        # Randomly select tournament members
        while len(tournament) < t_size:
            select = random.randint(0, len(population) - 1)
            tournament.append(population[select])
            t_fitness.append(fitnesses[select])

        while len(tournament) >= 1:

            # Member with highest fitness will be selected 
            # with probability of t_prob
            if random.random() < t_prob:
                parents.append(tournament[np.argmax(t_fitness)])
                break

            # Last remaining member of tournament will be selected
            elif len(tournament) == 1:
                parents.append(tournament[0])
                break

            # If member with highest fitness was not selected, 
            # then it is removed from tournament
            else:
                tournament.remove(tournament[np.argmax(t_fitness)])
                t_fitness.remove(t_fitness[np.argmax(t_fitness)])

    return parents


# Roulette wheel selection
def roulette(population, fitnesses):
    parents = []
    norm_fitnesses = normalize(fitnesses)
    return wheel(population, norm_fitnesses)


# Rank selection
def rank(population, fitnesses):

    # Initialization
    temp_population = population[:]
    temp_fitnesses = fitnesses[:]
    ranked_population = []
    ranked_fitnesses = []
    ranks = []
    probabilities = []
    rank = 1

    # Sort population and fitness lists
    while len(ranked_population) < len(population):
        index = np.argmin(temp_fitnesses)
        ranked_fitnesses.append(min(temp_fitnesses))
        del temp_fitnesses[index]
        ranked_population.append(temp_population[index])
        del temp_population[index]
        ranks.append(rank)
        rank += 1

    # Calculate probabilities
    for rank in ranks:
        probabilities.append(rank / (len(ranks) * (len(ranks) - 1)))

    return wheel(ranked_population, probabilities)


# Normalize fitness scores
def normalize(fitnesses):
    normalized = []
    total = sum(fitnesses)
    for fitness in fitnesses:
        normalized.append(fitness / total)
    return normalized


# Generate roulette wheel according to probabilities and select parents
def wheel(population, probabilities):

    # Initialization
    wheel = []
    parents = []
    value = 0

    # Generate roulette wheel
    for probability in probabilities:
        value += probability
        wheel.append(value)

    # Select parents from wheel
    while len(parents) < 2:
        select = random.random()
        print(select)
        for i, slot in enumerate(wheel):
            if select < slot:
                parents.append(population[i])
                break

    return parents
