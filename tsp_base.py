"""
tsp_base.py - Shared utilities for TSP solvers.

Provides coordinate loading, distance calculation, and result saving
so individual algorithm files stay focused on their core logic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def load_coords(csv_path: str, device: str) -> tuple[torch.Tensor, pd.DataFrame]:
    """Load city coordinates from a CSV file.

    Args:
        csv_path: Path to the CSV file (no header, two columns: x, y).
        device:   PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        A tuple of ``(coords_tensor, dataframe)`` where *coords_tensor* is
        a ``(N, 2)`` float32 tensor on *device* and *dataframe* is the raw
        pandas DataFrame (used later for index-based coordinate lookup).

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError:        If the file cannot be parsed as a numeric matrix.
    """
    try:
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        raise FileNotFoundError(f"좌표 파일을 찾을 수 없습니다: '{csv_path}'")
    except Exception as exc:
        raise ValueError(f"좌표 파일 파싱 오류 ({csv_path}): {exc}") from exc

    try:
        coords = torch.tensor(df.values, dtype=torch.float32).to(device)
    except Exception as exc:
        raise ValueError(f"좌표 텐서 변환 오류: {exc}") from exc

    return coords, df


def calculate_total_distance(paths: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Compute the total round-trip distance for a batch of TSP routes.

    Uses Euclidean distance between consecutive cities, wrapping around from
    the last city back to the first.

    Args:
        paths:  Integer tensor of shape ``(batch, num_cities)`` containing
                city indices for each route in the batch.
        coords: Float tensor of shape ``(num_cities, 2)`` with city coordinates.

    Returns:
        Float tensor of shape ``(batch,)`` with the total distance per route.
    """
    ordered_coords = coords[paths]                          # (batch, N, 2)
    rolled_coords = torch.roll(ordered_coords, shifts=-1, dims=1)
    deltas = ordered_coords - rolled_coords
    distances = (deltas ** 2).sum(dim=2).sqrt().sum(dim=1)  # (batch,)
    return distances


def save_solution(
    route: np.ndarray,
    df: pd.DataFrame,
    csv_path: str,
    png_path: str,
    best_dist: float,
    plot_color: str = "cyan",
    title_prefix: str = "TSP",
) -> None:
    """Persist the best route as a CSV and a PNG visualisation.

    The function also rotates the route so that city 0 appears first.

    Args:
        route:        1-D numpy array of city indices (the best route found).
        df:           Original DataFrame used to look up coordinates by index.
        csv_path:     Destination path for the solution CSV file.
        png_path:     Destination path for the route visualisation PNG.
        best_dist:    Best total distance (used in the plot title).
        plot_color:   Matplotlib colour string for the route line.
        title_prefix: Short label prepended to the plot title.

    Raises:
        OSError: If either output file cannot be written.
    """
    # Rotate so city 0 is first
    if 0 in route:
        zero_idx = np.where(route == 0)[0][0]
        route = np.concatenate((route[zero_idx:], route[:zero_idx]))

    solution_coords = df.values[route]

    # Save CSV
    try:
        pd.DataFrame(solution_coords).to_csv(csv_path, header=None, index=False)
        print(f"-> CSV 저장 완료: {csv_path}")
    except OSError as exc:
        raise OSError(f"CSV 저장 실패 ({csv_path}): {exc}") from exc

    # Save PNG visualisation
    try:
        path_plot = np.vstack([solution_coords, solution_coords[0]])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(path_plot[:, 0], path_plot[:, 1], color=plot_color,
                linewidth=0.5, alpha=0.8)
        ax.scatter(df.values[:, 0], df.values[:, 1], c="blue", s=5, alpha=0.5)
        ax.scatter(path_plot[0, 0], path_plot[0, 1], c="red",
                   marker="*", s=200, zorder=10, label="Start")
        ax.set_title(f"{title_prefix} Optimized TSP (Dist: {best_dist:.4f})")
        ax.legend()
        fig.savefig(png_path)
        plt.close(fig)
        print(f"-> PNG 저장 완료: {png_path}")
    except OSError as exc:
        raise OSError(f"PNG 저장 실패 ({png_path}): {exc}") from exc
