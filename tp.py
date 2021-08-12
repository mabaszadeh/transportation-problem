from itertools import permutations
from sys import maxsize
import numpy as np
from random import randrange

from inputs.in1 import *
# -------------------------------------------

cost = np.array(cost)
# -------------------------------------------


def north_west_corner(src, des):
    supply = list(map(supplies.get, src))
    demand = list(map(demands.get, des))

    i, j, fs = 0, 0, []

    while supply[len(supply) - 1] != 0:
        s, d = supply[i], demand[j]
        v = min(s, d)
        fs.append([[src[i], des[j]], v])

        supply[i] -= v
        demand[j] -= v

        if supply[i] == 0 and i < len(supply) - 1:
            i += 1
        if demand[j] == 0 and j < len(demand) - 1:
            j += 1

    return fs


def fitness(sol, cost):
    return np.sum(np.multiply(cost, sol))
# -------------------------------------------


population = None
optimal_solution = None
minimum_cost = maxsize
first_iteration = True

sources = list(supplies.keys())
destinations = list(demands.keys())
sol_list = []
counter = 0

# Creating initial population
for i in permutations(supplies):
    for j in permutations(demands):

        feasible_solution = north_west_corner(i, j)

        solution = np.zeros([len(supplies), len(demands)], dtype=int)

        for k in feasible_solution:
            solution[sources.index(k[0][0]),
                     destinations.index(k[0][1])] = k[1]

        if first_iteration:
            first_iteration = False

            ft = fitness(solution, cost)
            optimal_solution = solution
            minimum_cost = ft

            population = np.array([[solution, ft]])
            sol_list.append(np.array_str(solution))

        elif np.array_str(solution) not in sol_list:
            sol_list.append(np.array_str(solution))

            ft = fitness(solution, cost)
            if ft < minimum_cost:
                optimal_solution = solution
                minimum_cost = ft

            population = np.vstack([population, [solution, ft]])

        counter += 1
        if counter == MAX_POPULATION:
            break

    if counter == MAX_POPULATION:
        break
# -------------------------------------------


# lets free some memory ;)
del sol_list, sources, destinations
# -------------------------------------------


def selection():
    global population

    population = population[np.argsort(population[:, 1])]

    # Truncation
    population = population[:MAX_POPULATION]

    # Parent Selection: Random Selection
    np.random.shuffle(population)
# -------------------------------------------


def solution_repair(sol, demand):

    des_num = len(sol[0])
    offset = [0] * des_num

    tmp = np.sum(sol, axis=0)
    for i in range(des_num):
        offset[i] = tmp[i] - demand[i]

    for i in range(des_num):
        if offset[i] < 0:

            need = True
            j = 0

            while need and (j < des_num):
                if offset[j] > 0:

                    k = 0
                    has_extra = True

                    while k < len(sol) and has_extra:

                        val = min(offset[j], sol[k][j], abs(offset[i]))

                        sol[k][j] -= val
                        sol[k][i] += val

                        offset[i] += val
                        offset[j] -= val

                        if offset[j] == 0:
                            has_extra = False
                        if offset[i] == 0:
                            need = False

                        k += 1

                j += 1


def crossover():
    global optimal_solution, minimum_cost, population

    pool = np.empty((0, 2), dtype=int)

    if len(population) % 2 == 1:
        num = len(population) - 1
    else:
        num = len(population)

    for i in range(0, num, 2):
        if np.random.rand() <= CROSSOVER_RATE:

            child1 = population[i][0].copy()
            child2 = population[i+1][0].copy()

            comparison = (child1 == child2)
            if comparison.all():
                continue

            demand = np.sum(child1, axis=0)

            crossover_row = randrange(len(child1))

            t1 = child1[crossover_row].copy()
            t2 = child2[crossover_row].copy()
            child1[crossover_row] = t2
            child2[crossover_row] = t1

            solution_repair(child1, demand)
            solution_repair(child2, demand)

            ft = fitness(child1, cost)
            if ft < minimum_cost:
                optimal_solution = child1
                minimum_cost = ft

            pool = np.vstack([pool, [child1, ft]])

            ft = fitness(child2, cost)
            if ft < minimum_cost:
                optimal_solution = child2
                minimum_cost = ft

            pool = np.vstack([pool, [child2, ft]])

    population = np.vstack([population, pool])
# -------------------------------------------


def random_submatrix(mtx):
    tmp = mtx[len(mtx)-1:]

    i, j = np.where(tmp != 0)

    ti = randrange(len(i))

    SE = (i[ti] + 1, j[ti])

    # because randrange(0) is false
    if SE[1] == 0:
        NW = (randrange(SE[0]), 0)
    else:
        NW = (randrange(SE[0]), randrange(SE[1]))

    NE = (NW[0], SE[1])
    SW = (SE[0], NW[1])

    return NW, NE, SW, SE


def mutation():
    global optimal_solution, minimum_cost, population

    pool = np.empty((0, 2), dtype=int)

    for i in range(0, len(population)):
        if np.random.rand() <= MUTATION_RATE:

            solution = population[i][0].copy()

            NW, NE, SW, SE = random_submatrix(solution)

            val = min(solution[NW[0]][NW[1]], solution[SE[0]][SE[1]])

            solution[NW[0]][NW[1]] -= val
            solution[SE[0]][SE[1]] -= val
            solution[NE[0]][NE[1]] += val
            solution[SW[0]][SW[1]] += val

            ft = fitness(solution, cost)
            if ft < minimum_cost:
                optimal_solution = solution
                minimum_cost = ft

            pool = np.vstack([pool, [solution, ft]])

    population = np.vstack([population, pool])
# -------------------------------------------


for i in range(ITERATION_NUM):
    selection()
    crossover()
    mutation()
# -------------------------------------------


print("\nMinimum cost:", minimum_cost)
print()
print("Optimal solution:\n")
print(optimal_solution)
