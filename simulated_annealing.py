"""
simulated_annealing.py - GPU-accelerated Parallel Simulated Annealing for TSP.

Shared I/O utilities come from tsp_base.py. This file owns only the
SA-specific logic: temperature schedule, Metropolis acceptance, and the
main annealing loop.
"""

import time

import numpy as np
import torch

from tsp_base import load_coords, calculate_total_distance, save_solution

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 4096      # Parallel routes explored simultaneously
                             # (reduce to 2048 if GPU memory is insufficient)
STEPS: int = 150_000        # Total annealing steps
START_TEMP: float = 20.0    # Initial temperature (tuned for normalised coords)
COOLING_RATE: float = 0.9997  # Multiplicative cooling factor per step
MIN_TEMP: float = 0.0001    # Early-stop threshold: stop if temp drops below this
LOG_INTERVAL: int = 5_000   # Print progress every N steps

DATA_CSV: str = "2024_AI_TSP.csv"
OUTPUT_CSV: str = "solution_SA.csv"
OUTPUT_PNG: str = "tsp_SA_result.png"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

print("=== GPU Parallel SA 시작 ===")
print(f"장치: {DEVICE} | 동시 탐색: {BATCH_SIZE}개 | 총 반복: {STEPS}회")
print("-" * 61)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
coords, df = load_coords(DATA_CSV, DEVICE)
num_cities: int = len(coords)

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
current_paths = torch.argsort(torch.rand(BATCH_SIZE, num_cities, device=DEVICE), dim=1)
current_dists = calculate_total_distance(current_paths, coords)

best_global_dist: float = current_dists.min().item()
best_global_path: torch.Tensor = current_paths[current_dists.argmin()].clone()

start_time = time.time()
temp: float = START_TEMP

print(f"[초기 상태] 최단 거리: {best_global_dist:.4f}")
print("-" * 61)

# ---------------------------------------------------------------------------
# Main annealing loop
# ---------------------------------------------------------------------------
for step in range(1, STEPS + 1):
    # (1) Generate neighbours via random swap mutation
    new_paths = current_paths.clone()

    idx1 = torch.randint(0, num_cities, (BATCH_SIZE,), device=DEVICE)
    idx2 = torch.randint(0, num_cities, (BATCH_SIZE,), device=DEVICE)
    batch_indices = torch.arange(BATCH_SIZE, device=DEVICE)

    val1 = new_paths[batch_indices, idx1]
    val2 = new_paths[batch_indices, idx2]
    new_paths[batch_indices, idx1] = val2
    new_paths[batch_indices, idx2] = val1

    # (2) Metropolis acceptance criterion
    new_dists = calculate_total_distance(new_paths, coords)
    diff = current_dists - new_dists
    probs = torch.exp(diff / temp)
    accept_mask = (diff > 0) | (torch.rand(BATCH_SIZE, device=DEVICE) < probs)

    current_paths[accept_mask] = new_paths[accept_mask]
    current_dists[accept_mask] = new_dists[accept_mask]

    # (3) Update global best
    min_dist_batch, min_idx = current_dists.min(dim=0)
    if min_dist_batch.item() < best_global_dist:
        best_global_dist = min_dist_batch.item()
        best_global_path = current_paths[min_idx].clone()

    # (4) Cool down
    temp *= COOLING_RATE

    # Progress reporting
    if step % LOG_INTERVAL == 0 or step == STEPS:
        elapsed = time.time() - start_time
        progress = (step / STEPS) * 100
        print(
            f"[{progress:5.1f}%] 단계: {step:6d} | 온도: {temp:7.4f} | "
            f"현재 최단 거리: {best_global_dist:.4f} | 경과: {elapsed:.1f}초"
        )

    if temp < MIN_TEMP:
        print("\n온도가 너무 낮아져 조기 종료합니다.")
        break

# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------
total_time = time.time() - start_time
print("-" * 61)
print(f"탐색 완료! 총 소요 시간: {total_time:.2f}초")
print(f"최종 최단 거리: {best_global_dist:.4f}")

final_route: np.ndarray = best_global_path.cpu().numpy()

save_solution(
    route=final_route,
    df=df,
    csv_path=OUTPUT_CSV,
    png_path=OUTPUT_PNG,
    best_dist=best_global_dist,
    plot_color="green",
    title_prefix="SA",
)
