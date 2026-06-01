import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def initialization(pop_size, dim, ub, lb):
    """
    Initialize population within bounds
    """
    ub = np.array(ub)
    lb = np.array(lb)

    if ub.size == 1:
        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

    X = np.random.rand(pop_size, dim) * (ub - lb) + lb
    return X


def PKO(pop_size, max_iteration, lb, ub, dim, fobj):
    start_time = time.time()

    BF = 8  # Beating Factor
    crest_angles = 2 * np.pi * np.random.rand()

    X = initialization(pop_size, dim, ub, lb)

    fitness = np.zeros(pop_size)
    convergence_curve = np.zeros(max_iteration)

    # Initial fitness
    for i in range(pop_size):
        fitness[i] = fobj(X[i, :])

    # Best solution
    sorted_indexes = np.argsort(fitness)
    best_position = X[sorted_indexes[0], :].copy()
    best_fitness = fitness[sorted_indexes[0]]

    convergence_curve[0] = best_fitness

    t = 1

    PEmax = 0.5
    PEmin = 0
    while t < max_iteration + 1:
        
        o = np.exp(-t / max_iteration) ** 2

        X1 = np.copy(X)
        print('iteration...........',t)

        # =========================
        # Exploration / Exploitation
        # =========================
        for i in range(pop_size):

            # Exploration
            if np.random.rand() < 0.8:

                j = i
                while i == j:
                    j = np.random.randint(0, pop_size)

                beating_rate = np.random.rand() * (fitness[j] / fitness[i])

                alpha = 2 * np.random.randn(dim) - 1

                if np.random.rand() < 0.5:

                    T = beating_rate - (
                        (t ** (1 / BF)) / (max_iteration ** (1 / BF))
                    )

                    X1[i, :] = X[i, :] + alpha * T * (X[j, :] - X[i, :])

                else:

                    T = (
                        (np.exp(1) - np.exp(((t - 1) / max_iteration) ** (1 / BF)))
                        * np.cos(crest_angles)
                    )

                    X1[i, :] = X[i, :] + alpha * T * (X[j, :] - X[i, :])

            # Exploitation
            else:

                alpha = 2 * np.random.randn(dim) - 1

                b = X[i, :] + (o ** 2) * np.random.randn(dim) * best_position

                hunting_ability = np.random.rand() * (
                    fitness[i] / best_fitness
                )

                X1[i, :] = (
                    X[i, :]
                    + hunting_ability * o * alpha * (b - best_position)
                )

        # =========================
        # Boundary check and update
        # =========================
        for i in range(pop_size):

            X1[i, :] = np.clip(X1[i, :], lb, ub)

            fitness_new = fobj(X1[i, :])

            if fitness_new < fitness[i]:
                fitness[i] = fitness_new
                X[i, :] = X1[i, :]

            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best_position = X[i, :].copy()

        # ==================================================
        # Commensal association with Eurasian otters
        # ==================================================
        PE = PEmax - (PEmax - PEmin) * (t / max_iteration)

        for i in range(pop_size):

            alpha = 2 * np.random.randn(dim) - 1

            if np.random.rand() > (1 - PE):

                rand1 = np.random.randint(0, pop_size)
                rand2 = np.random.randint(0, pop_size)

                X1[i, :] = (
                    X[rand1, :]
                    + o * alpha * np.abs(X[i, :] - X[rand2, :])
                )

            else:
                X1[i, :] = X[i, :]

            # Boundary control
            X1[i, :] = np.clip(X1[i, :], lb, ub)

            fitness_new = fobj(X1[i, :])

            if fitness_new < fitness[i]:
                fitness[i] = fitness_new
                X[i, :] = X1[i, :]

            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best_position = X[i, :].copy()

        convergence_curve[t - 1] = best_fitness

        t += 1

    total_time = time.time() - start_time

    return   best_fitness, best_position,  convergence_curve
    


