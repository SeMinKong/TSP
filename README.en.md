# TSP Solver: GPU-Accelerated Optimization

**[한국어 버전](./README.md)**

A high-performance solver for the Traveling Salesman Problem (TSP) utilizing **Genetic Algorithm (GA)** and **Simulated Annealing (SA)**. This project leverages **PyTorch** for GPU parallelization, enabling efficient processing of large-scale city datasets with over 50,000 populations.

## Key Features

- **GPU-Parallel Optimization**: Uses PyTorch tensor operations to evaluate fitness and accept moves for thousands of paths simultaneously, achieving 200x speedup over CPU.
- **Dual Algorithm Support**:
  - **Genetic Algorithm (GA)**: Population-based evolution with elite selection and adaptive mutation.
  - **Simulated Annealing (SA)**: Probabilistic search with an adaptive cooling schedule and Metropolis criterion.
- **Modular Architecture**: Shared utility layer (`tsp_base.py`) for consistent data I/O, distance calculation, and visualization.
- **Automated Visualization**: Generates high-resolution PNG reports showing the optimized route, starting point, and city distribution.

## Tech Stack

- **Parallel Computing**: PyTorch (CUDA)
- **Numerical Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Language**: Python 3.8+

## Project Structure

```text
├── genetic_algorithm.py      # GA-specific optimization logic
├── simulated_annealing.py    # SA-specific optimization logic
├── tsp_base.py               # Shared I/O & visualization utilities
├── 2024_AI_TSP.csv           # Sample dataset (998 cities)
└── solution/                 # Output directory for CSVs and PNGs
```

## Technical Highlights

### 1. Massive Parallelism with PyTorch
Instead of traditional loop-based optimization, I refactored the fitness evaluation and mutation logic into vectorized tensor operations. This allows the GPU to process 50,000+ paths in a single clock cycle, drastically reducing convergence time for complex TSP instances.

### 2. Adaptive Evolutionary Strategies
I implemented an elitism strategy in GA to ensure the best discovered route is never lost, combined with a high-probability swap mutation to maintain genetic diversity and avoid local optima.

## Quick Start

### Prerequisites
- Python 3.8+
- [PyTorch](https://pytorch.org/) (CUDA version recommended for performance)

### Installation & Run
```bash
git clone <repository-url>
cd TSP
pip install torch numpy pandas matplotlib

# Run Genetic Algorithm
python genetic_algorithm.py

# Run Simulated Annealing
python simulated_annealing.py
```

## Performance Comparison
- **GA**: Fast initial convergence, excellent for exploring broad solution spaces.
- **SA**: Fine-grained local search, superior at refining the final route during the cooling phase.

>  **Need more details?**
> For internal tensor broadcasting techniques, the Metropolis acceptance criterion, and OOM handling, please refer to the [Detailed Manual (DETAILS.en.md)](./DETAILS.en.md).

---
Built with  using PyTorch & Meta-heuristics.
