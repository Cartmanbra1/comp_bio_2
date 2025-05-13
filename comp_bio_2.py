# Roie Amsalem 322535436
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import copy

# Constants
N = 5  # For Most Perfect Magic Square, use N = 4, 8, 12, ...
MAGIC_SUM = N * (N**2 + 1) // 2
POPULATION_SIZE = 100
MAX_GENERATIONS = 500
MUTATION_RATE = 0.2
ELITISM_COUNT = 5
OPTIMIZATION_STEPS = 5
NO_IMPROVEMENT_LIMIT = 50
fitness_calls = 0

def flatten(matrix: List[List[int]]) -> List[int]:
    """
    Flattens a 2D matrix into a 1D list.

    Args:
        matrix (List[List[int]]): A 2D matrix.

    Returns:
        List[int]: A flattened 1D list of the matrix elements.
    """
    return [item for row in matrix for item in row]

def unflatten(lst: List[int]) -> List[List[int]]:
    """
    Converts a 1D list into a 2D N x N matrix.

    Args:
        lst (List[int]): A list of N*N integers.

    Returns:
        List[List[int]]: A 2D matrix representation of the list.
    """
    return [lst[i * N:(i + 1) * N] for i in range(N)]

def is_most_perfect() -> bool:
    """
    Checks if N is a multiple of 4, indicating the need for
    Most Perfect Magic Square conditions.

    Returns:
        bool: True if N is divisible by 4, False otherwise.
    """
    return N % 4 == 0

def evaluate_fitness(square: List[List[int]]) -> int:
    """
    Evaluates the fitness of a square by calculating deviation
    from the magic sum across rows, columns, diagonals, and,
    if applicable, additional Most Perfect Magic Square conditions.

    Args:
        square (List[List[int]]): A candidate magic square.

    Returns:
        int: The fitness score (lower is better).
    """
    global fitness_calls
    fitness_calls += 1
    fitness = 0

    for row in square:
        fitness += abs(sum(row) - MAGIC_SUM)
    for col in zip(*square):
        fitness += abs(sum(col) - MAGIC_SUM)
    diag1 = sum(square[i][i] for i in range(N))
    diag2 = sum(square[i][N - 1 - i] for i in range(N))
    fitness += abs(diag1 - MAGIC_SUM)
    fitness += abs(diag2 - MAGIC_SUM)

    if is_most_perfect():
        for i in range(N - 1):
            for j in range(N - 1):
                block_sum = square[i][j] + square[i+1][j] + square[i][j+1] + square[i+1][j+1]
                fitness += abs(block_sum - MAGIC_SUM)
        for i in range(N):
            for j in range(N // 2):
                fitness += abs((square[i][j] + square[i][N - 1 - j]) - (N*N + 1))
                fitness += abs((square[j][i] + square[N - 1 - j][i]) - (N*N + 1))

    return fitness

def create_individual() -> List[List[int]]:
    """
    Generates a random valid square containing all integers from 1 to N^2.

    Returns:
        List[List[int]]: A randomly shuffled magic square candidate.
    """
    lst = list(range(1, N*N + 1))
    random.shuffle(lst)
    return unflatten(lst)

def mutate(square: List[List[int]]) -> List[List[int]]:
    """
    Performs a mutation by swapping two random elements in the square.

    Args:
        square (List[List[int]]): The square to mutate.

    Returns:
        List[List[int]]: A new square with two elements swapped.
    """
    flat = flatten(square)
    i, j = random.sample(range(N*N), 2)
    flat[i], flat[j] = flat[j], flat[i]
    return unflatten(flat)

def crossover(parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
    """
    Performs single-point crossover between two parents to produce a child.

    Args:
        parent1 (List[List[int]]): The first parent square.
        parent2 (List[List[int]]): The second parent square.

    Returns:
        List[List[int]]: A new child square.
    """
    flat1, flat2 = flatten(parent1), flatten(parent2)
    child = [0] * (N * N)
    point = random.randint(0, N*N - 1)
    used = set()
    for i in range(point):
        child[i] = flat1[i]
        used.add(flat1[i])
    idx = point
    for val in flat2:
        if val not in used:
            child[idx] = val
            used.add(val)
            idx += 1
            if idx >= N*N:
                break
    return unflatten(child)

def local_optimization(square: List[List[int]]) -> List[List[int]]:
    """
    Improves a square locally using OPTIMIZATION_STEPS of mutations.

    Args:
        square (List[List[int]]): The square to optimize.

    Returns:
        List[List[int]]: A potentially improved square.
    """
    best = copy.deepcopy(square)
    best_fitness = evaluate_fitness(best)
    for _ in range(OPTIMIZATION_STEPS):
        candidate = mutate(copy.deepcopy(best))
        fit = evaluate_fitness(candidate)
        if fit < best_fitness:
            best = candidate
            best_fitness = fit
    return best

def run_evolution(strategy="classic") -> Tuple[List[int], List[int], List[List[int]]]:
    """
    Runs the evolutionary algorithm with a specified strategy.

    Args:
        strategy (str): One of 'classic', 'darwinian', or 'lamarckian'.

    Returns:
        Tuple[List[int], List[int], List[List[int]]]: Lists of best scores,
        average scores, and the best solution found.
    """
    global fitness_calls
    fitness_calls = 0
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_scores = []
    avg_scores = []
    best_fitness_seen = float('inf')
    stagnant_generations = 0

    for gen in range(MAX_GENERATIONS):
        if strategy == "darwinian":
            evaluated = [(evaluate_fitness(local_optimization(ind)), ind) for ind in population]
        elif strategy == "lamarckian":
            population = [local_optimization(ind) for ind in population]
            evaluated = [(evaluate_fitness(ind), ind) for ind in population]
        else:
            evaluated = [(evaluate_fitness(ind), ind) for ind in population]

        evaluated.sort(key=lambda x: x[0])
        current_best_fitness = evaluated[0][0]
        best_scores.append(current_best_fitness)
        avg_scores.append(sum(f for f, _ in evaluated) / POPULATION_SIZE)

        if current_best_fitness == 0:
            break
        if current_best_fitness < best_fitness_seen:
            best_fitness_seen = current_best_fitness
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        if stagnant_generations >= NO_IMPROVEMENT_LIMIT:
            break

        new_population = [copy.deepcopy(evaluated[i][1]) for i in range(ELITISM_COUNT)]
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = random.sample(evaluated[:20], 2)
            child = crossover(p1[1], p2[1])
            if child not in new_population:
                if random.random() < MUTATION_RATE or strategy == "classic":
                    child = mutate(child)
                new_population.append(child)
        population = new_population

    return best_scores, avg_scores, evaluated[0][1]


# Run all strategies
results = {}
for strategy in ["classic", "darwinian", "lamarckian"]:
    best_scores, avg_scores, best_solution = run_evolution(strategy)
    results[strategy] = {
        "best_scores": best_scores,
        "avg_scores": avg_scores,
        "solution": best_solution
    }

# Plot results
plt.figure(figsize=(10, 6))
for strategy in results:
    plt.plot(results[strategy]["best_scores"], label=f"{strategy} best")
    plt.plot(results[strategy]["avg_scores"], linestyle="--", label=f"{strategy} avg")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title(f"GA Comparison (N={N}) - Including Most Perfect Magic Square & Stagnation Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Show best solutions
for strategy in results:
    print(f"\nBest solution for {strategy} (fitness {evaluate_fitness(results[strategy]['solution'])}):")
    for row in results[strategy]["solution"]:
        print(row)