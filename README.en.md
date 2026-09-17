# TSP Solver: GPU-Accelerated Optimization

**[한국어 버전](./README.md)**

I built this Traveling Salesman Problem (TSP) solver with a **Genetic Algorithm (GA)** and **Simulated Annealing (SA)**. PyTorch tensor operations evaluate tens of thousands of candidate routes on the GPU at once.

## Key Features

- **GPU-Parallel Optimization**: Evaluates populations of more than 50,000 candidates with PyTorch. It was about 200 times faster than the previous CPU code.
- **Dual Algorithm Support**:
  - **Genetic Algorithm (GA)**: Population-based evolution with elite selection and adaptive mutation.
  - **Simulated Annealing (SA)**: Probabilistic search with an adaptive cooling schedule and Metropolis criterion.
- **Modular Architecture**: Shared utility layer (`tsp_base.py`) for consistent data I/O, distance calculation, and visualization.
- **Result Visualization**: Saves the calculated route, starting point, and city distribution as a PNG.

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
I rewrote fitness evaluation and mutation as vectorized tensor operations instead of per-route loops. This lets the GPU calculate many route lengths together.

### 2. Adaptive Evolutionary Strategies
The GA carries the top 20% into the next generation and applies adaptive mutation. The SA implementation uses the Metropolis criterion and a cooling schedule to decide whether to accept a candidate route.

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

Tensor broadcasting, the Metropolis acceptance criterion, and GPU memory handling are documented in the [detailed manual](./DETAILS.en.md).
