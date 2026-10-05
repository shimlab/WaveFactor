"""Simple multi-tier performance benchmark for WaveFactor."""

import argparse
import os
import sys
import time
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wavefactor.data import prepare_spatial_data
from wavefactor.engine import init_parameters, build_parameters_cpp, _get_cpp_backend

cpp = _get_cpp_backend()


def run_tier(name: str, grid_side: int, n_genes: int, K: int, D: int, iters: int = 10, n_iter1_runs: int = 3):
    n_spots = grid_side * grid_side
    print(f"--- {name}: {n_spots} spots ({grid_side}x{grid_side}), {n_genes} genes, K={K}, D={D} ---")

    # 1. Benchmark data preparation (excluded from pipeline timing)
    coords = np.column_stack([
        np.repeat(np.arange(grid_side), grid_side),
        np.tile(np.arange(grid_side), grid_side),
    ])
    rng = np.random.RandomState(42)
    X = rng.randn(n_spots, n_genes)

    # 2. Measure one-off setup (2D DWT + marshalling)
    t0_prep = time.perf_counter()
    data = prepare_spatial_data(X=X, coords=coords, n_factors=K, n_length_scales=D)
    params = init_parameters(Y=data.true_Y, dimensions=data.dimensions, rng=rng)
    cpp_params = build_parameters_cpp(params, cpp)
    t_prep = time.perf_counter() - t0_prep

    # 3. Measure full CAVI compute loop (standard run)
    t0_cavi = time.perf_counter()
    result = cpp.cavi(cpp_params, iters, 0.0)  # Use tol=0.0 so all iters run
    t_cavi = time.perf_counter() - t0_cavi

    actual_iters = getattr(result, "iterations", iters)
    time_per_iter = t_cavi / actual_iters

    # 4. Measure Iteration 1 across multiple trials to isolate initial overhead
    iter1_times = []
    for _ in range(n_iter1_runs):
        p_fresh = init_parameters(Y=data.true_Y, dimensions=data.dimensions, rng=rng)
        cpp_p_fresh = build_parameters_cpp(p_fresh, cpp)
        
        t0_single = time.perf_counter()
        _ = cpp.cavi(cpp_p_fresh, 1, 0.0)
        iter1_times.append(time.perf_counter() - t0_single)

    avg_iter1 = float(np.mean(iter1_times))

    # 5. Delta calculation excluding the initial iteration
    if actual_iters > 1:
        steady_state_per_iter = max(0.0, (t_cavi - avg_iter1) / (actual_iters - 1))
    else:
        steady_state_per_iter = time_per_iter

    print(f"  Prep / DWT time:                 {t_prep:.3f} s (one-off)")
    print(f"  CAVI C++ time:                   {t_cavi:.3f} s  ({time_per_iter:.4f} s / iter over {actual_iters} iters)")
    print(f"  Avg Iteration 1 time ({n_iter1_runs} runs):  {avg_iter1:.4f} s")
    print(f"  Estimated CAVI / Iter (excl. 1): {steady_state_per_iter:.4f} s / iter")
    print(f"  Total run time:                  {t_prep + t_cavi:.3f} s\n")

    return (name, n_spots, n_genes, K, time_per_iter, steady_state_per_iter)


def print_summary(results):
    """Print benchmark summary table."""
    print("=" * 76)
    print(f"{'Tier':<19}| {'Spots':<8}| {'Genes':<7}| {'K':<4}| CAVI / Iter | Excl. Iter 1")
    print("-" * 76)
    for name, spots, genes, k, per_iter, steady_per_iter in results:
        print(f"{name:<19}| {spots:<8}| {genes:<7}| {k:<4}| {per_iter:.4f} s    | {steady_per_iter:.4f} s")
    print("=" * 76)


if __name__ == "__main__":
    if cpp is None:
        raise RuntimeError("WaveFactor C++ backend not found. Compile the C++ extension first.")

    parser = argparse.ArgumentParser(description="Run WaveFactor benchmarks.")
    has_cli_args = len(sys.argv) > 1

    parser.add_argument("--grid_side", type=int, required=has_cli_args, help="Grid side length")
    parser.add_argument("--n_genes", type=int, required=has_cli_args, help="Number of genes")
    parser.add_argument("-K", type=int, required=has_cli_args, help="Number of factors")
    parser.add_argument("-D", type=int, required=has_cli_args, help="Number of length scales")
    parser.add_argument("--iters", type=int, required=has_cli_args, help="Number of CAVI iterations")
    parser.add_argument("--n_iter1_runs", type=int, required=has_cli_args, help="Number of iteration 1 runs")

    args = parser.parse_args()

    if has_cli_args:
        print("Running Single Tier Performance Benchmark for WaveFactor...\n")
        results = [
            run_tier(
                "Single Tier",
                grid_side=args.grid_side,
                n_genes=args.n_genes,
                K=args.K,
                D=args.D,
                iters=args.iters,
                n_iter1_runs=args.n_iter1_runs,
            )
        ]
    else:
        print("Running Multi-Tier Performance Benchmark for WaveFactor...\n")
        results = [
            run_tier("Tier 1 (Small)",  grid_side=16,  n_genes=1000, K=10, D=4, iters=10),
            run_tier("Tier 2 (Medium)", grid_side=32,  n_genes=1000, K=10, D=5, iters=10),
            run_tier("Tier 3 (Large)",  grid_side=64,  n_genes=1000, K=10, D=6, iters=10),
            run_tier("Tier 4 (XLarge)", grid_side=128, n_genes=1000, K=10, D=7, iters=10),
        ]

    print_summary(results)