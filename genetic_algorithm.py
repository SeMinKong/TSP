"""
genetic_algorithm.py - GPU-accelerated Genetic Algorithm for TSP.

Shared I/O utilities come from tsp_base.py. This file owns only the
GA-specific logic: population initialisation, crossover, mutation, and
the main evolution loop.
"""

import time

import numpy as np
import torch

from tsp_base import load_coords, calculate_total_distance, save_solution

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
POPULATION_SIZE: int = 50_000   # Number of candidate routes per generation
GENERATIONS: int = 10_000       # Total number of evolution steps
MUTATION_RATE: float = 0.05     # Base mutation rate (fraction of population)
ELITE_FRACTION: float = 0.20    # Top fraction preserved unchanged each generation
INNER_MUTATION_RATE: float = 0.40  # Mutation rate applied to offspring
LOG_INTERVAL: int = 100         # Print progress every N generations

DATA_CSV: str = "2024_AI_TSP.csv"
OUTPUT_CSV: str = "./solution/solution_GA.csv"
OUTPUT_PNG: str = "./solution/tsp_GA_result.png"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 장치: {DEVICE}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
coords, df = load_coords(DATA_CSV, DEVICE)
num_cities: int = len(coords)

# ---------------------------------------------------------------------------
# Algorithm functions
# ---------------------------------------------------------------------------

def init_population(pop_size: int, num_cities: int, device: str) -> torch.Tensor:
    """Return a random population of permutations.

    Args:
        pop_size:   Number of individuals.
        num_cities: Number of cities (permutation length).
        device:     PyTorch device string.

    Returns:
        Integer tensor of shape ``(pop_size, num_cities)``.
    """
    return torch.argsort(torch.rand(pop_size, num_cities, device=device), dim=1)


def crossover(parents: torch.Tensor) -> torch.Tensor:
    """Return offspring from parents.

    TSP-valid OX crossover is expensive to parallelise on GPU, so this
    implementation uses an elite-clone + mutation strategy instead: the
    function simply returns a clone of the parents and lets :func:`mutate`
    introduce diversity.

    Args:
        parents: Integer tensor of shape ``(pop_size, num_cities)``.

    Returns:
        Clone of *parents* with the same shape.
    """
    return parents.clone()


def mutate(pop: torch.Tensor, rate: float) -> torch.Tensor:
    """Apply random swap mutations in-place to a fraction of the population.

    Args:
        pop:  Integer tensor of shape ``(pop_size, num_cities)``.
        rate: Fraction of individuals to mutate (0.0–1.0).

    Returns:
        The mutated *pop* tensor (modified in-place and returned).
    """
    num_mutations = int(pop.shape[0] * rate)
    if num_mutations == 0:
        return pop

    target_indices = torch.randint(0, pop.shape[0], (num_mutations,), device=DEVICE)
    col1 = torch.randint(0, num_cities, (num_mutations,), device=DEVICE)
    col2 = torch.randint(0, num_cities, (num_mutations,), device=DEVICE)

    val1 = pop[target_indices, col1]
    val2 = pop[target_indices, col2]
    pop[target_indices, col1] = val2
    pop[target_indices, col2] = val1

    return pop


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------
population = init_population(POPULATION_SIZE, num_cities, DEVICE)

print(f"GPU 가속 시작! (Pop: {POPULATION_SIZE}, Gen: {GENERATIONS})")
start_time = time.time()

for i in range(GENERATIONS):
    # 1. Fitness evaluation (shorter distance = higher fitness)
    dists = calculate_total_distance(population, coords)

    # 2. Sort by distance (ascending)
    sorted_indices = torch.argsort(dists)
    population = population[sorted_indices]
    dists = dists[sorted_indices]

    if (i + 1) % LOG_INTERVAL == 0:
        print(f"Gen {i + 1}: Best Dist = {dists[0].item():.4f}")

    # 3. Next generation: elite preservation + mutation
    elite_count = int(POPULATION_SIZE * ELITE_FRACTION)
    elites = population[:elite_count]

    offsprings = elites.repeat(int(POPULATION_SIZE / elite_count), 1)
    offsprings = offsprings[:POPULATION_SIZE]

    population = mutate(crossover(offsprings), rate=INNER_MUTATION_RATE)

    # Always keep the single best solution unchanged
    population[0] = elites[0]

end_time = time.time()
print(f"완료! 소요 시간: {end_time - start_time:.2f}초")

# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------
best_route: np.ndarray = population[0].cpu().numpy()
best_dist: float = dists[0].item()
print(f"최종 최단 거리: {best_dist:.4f}")

save_solution(
    route=best_route,
    df=df,
    csv_path=OUTPUT_CSV,
    png_path=OUTPUT_PNG,
    best_dist=best_dist,
    plot_color="cyan",
    title_prefix="GA",
)
