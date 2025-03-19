from __future__ import annotations

from enum import IntEnum, Enum

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Base(IntEnum):
    A = 0,
    T = 1,
    C = 2,
    G = 3


class Action(Enum):
    Kill = 0.2
    Replicate = 0.4
    Survive = 0.4


class Individual:
    """
    Cell class. Holds all the basic data of cell.
    """
    gene_code: list[Base] = []
    parent = None
    generation: int

    def __init__(self, length: int | None, generation: int = 1):
        if length is not None:
            self.gene_code = np.random.choice([Base.A, Base.T, Base.C, Base.G], length)
        self.generation = generation

    def set_gene_code(self, gene_code):
        self.gene_code = gene_code

    def set_parent(self, parent):
        self.parent = parent


class Simulation:
    population_size: int
    population: list[list[Individual]] = []

    def __init__(self, length: int, population_size: int):
        self.population_size = population_size
        self.population = []
        first_generation = []
        # Populate first generation
        for i in range(population_size):
            first_generation.append(Individual(length, 1))
        self.population.append(first_generation)

    def run(self, iterations: int):
        """
        Base simulation for finite population dynamics
        :param iterations: Amount of iterations
        :return:
        """
        for i in range(iterations):
            # Sample parents from current generation
            parents: list[Individual] = np.random.choice(self.population[-1], self.population_size)
            new_population: list[Individual] = []
            # Create child from parent for new generation
            for parent in parents:
                I = Individual(None, parent.generation + 1)
                I.set_gene_code(parent.gene_code.copy())
                I.set_parent(parent)
                new_population.append(I)

            self.population.append(new_population)

    def common_ancestor(self, first: Individual, second: Individual) -> Individual | None:
        first_parents: list[Individual] = [first]
        second_parents: list[Individual] = [second]

        if first == second:
            return first

        common_ancestor_found: bool = False
        while not common_ancestor_found:
            f_p = []
            s_p = []

            for parent in first_parents:
                if parent.parent is None:
                    return None
                f_p.append(parent.parent)
            for parent in second_parents:
                if parent.parent is None:
                    return None
                s_p.append(parent.parent)

            for parent in f_p:
                if parent in s_p:
                    return parent

            first_parents = f_p
            second_parents = s_p

        return None

    def common_ancestor_whole(self, children: list[Individual]) -> Individual | None:
        common_ancestor_found: bool = False
        current_generation = children
        while not common_ancestor_found:
            parent_generation = set()

            # For current generation cells, get parents
            for c in current_generation:
                if c.parent is None:
                    return
                parent_generation.add(c.parent)

            # If only one parent => MRCA
            if len(parent_generation) == 1:
                return parent_generation.pop()

            current_generation = parent_generation

        return None

def plot_common_ancestor_hist(repeat: int):
    generations = []
    length = 300
    for i in range(repeat):
        S = Simulation(20, 50)
        S.run(length)
        generation = 0
        p = S.common_ancestor_whole(S.population[-1])
        if p is not None:
            generation += p.generation
        # Flip generation
        generations.append(length - generation)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title("Cell common ancestor generations difference")
    sns.histplot(generations, bins=50)
    ax.set_xlim([0, 200])
    plt.show()

def plot_population():
    S = Simulation(20, 50)
    S.run(400)
    cells = np.zeros((len(S.population), len(S.population[0])))
    for i in range(len(S.population)):
        for j in range(len(S.population[0])):
            cells[i, j] = sum(S.population[i][j].gene_code)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(cells, cmap='magma')
    ax.set_title("Cell culture over time")
    plt.gca().set_aspect(0.1)
    plt.show()

if __name__ == '__main__':
    plot_common_ancestor_hist(5000)
    plot_population()