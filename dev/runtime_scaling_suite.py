"""
WaveFactor scaling benchmark & Matplotlib visualization script.

Sweeps:
  1. grid_side: [8, 16, 32, 64, 128, 256, 512] (vs. quadratic ref)
  2. n_spots (grid_side^2): [8^2, ..., 512^2]
  3. n_genes:   [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
  4. K:         [5, 10, 15, 20, 25, 30, 35, 40]
  5. D:         [2, 3, 4, 5, 6]
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wavefactor.data import prepare_spatial_data
from wavefactor.engine import init_parameters, build_parameters_cpp, _get_cpp_backend

cpp = _get_cpp_backend()
if cpp is None:
    raise RuntimeError("WaveFactor C++ backend not found. Compile the C++ extension first.")


def benchmark_single(grid_side: int, n_genes: int, K: int, D: int, iters: int = 10, n_iter1_runs: int = 3) -> float:
    """Measures steady-state time per iteration (excluding Iteration 1)."""
    n_spots = grid_side * grid_side

    coords = np.column_stack([
        np.repeat(np.arange(grid_side), grid_side),
        np.tile(np.arange(grid_side), grid_side),
    ])
    rng = np.random.RandomState(42)
    X = rng.randn(n_spots, n_genes)

    # Setup
    data = prepare_spatial_data(X=X, coords=coords, n_factors=K, n_length_scales=D)
    params = init_parameters(Y=data.true_Y, dimensions=data.dimensions, rng=rng)
    cpp_params = build_parameters_cpp(params, cpp)

    # Full CAVI loop
    t0_cavi = time.perf_counter()
    result = cpp.cavi(cpp_params, iters, 0.0)
    t_cavi = time.perf_counter() - t0_cavi

    actual_iters = getattr(result, "iterations", iters)

    # Isolate iteration 1 overhead
    iter1_times = []
    for _ in range(n_iter1_runs):
        p_fresh = init_parameters(Y=data.true_Y, dimensions=data.dimensions, rng=rng)
        cpp_p_fresh = build_parameters_cpp(p_fresh, cpp)
        t0_single = time.perf_counter()
        _ = cpp.cavi(cpp_p_fresh, 1, 0.0)
        iter1_times.append(time.perf_counter() - t0_single)

    avg_iter1 = float(np.mean(iter1_times))
    if actual_iters > 1:
        steady_state_per_iter = max(0.0, (t_cavi - avg_iter1) / (actual_iters - 1))
    else:
        steady_state_per_iter = t_cavi / actual_iters

    return steady_state_per_iter


def run_all_benchmarks():
    # Baseline configuration
    baseline = {"grid_side": 64, "n_genes": 250, "K": 5, "D": 3, "iters": 100}

    # Sweep values
    sweep_grid = [8, 16, 32, 64, 128, 256, 512]
    sweep_genes = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    sweep_k = [5, 10, 15, 20, 25, 30, 35, 40]
    sweep_d = [2, 3, 4, 5, 6]

    print("=== 1/4: Benchmarking grid_side sweep ===")
    times_grid = []
    for gs in sweep_grid:
        print(f"  grid_side={gs} ({gs*gs} spots)...")
        t = benchmark_single(grid_side=gs, n_genes=baseline["n_genes"], K=baseline["K"], D=baseline["D"], iters=baseline["iters"])
        times_grid.append(t)

    print("\n=== 2/4: Benchmarking n_genes sweep ===")
    times_genes = []
    for g in sweep_genes:
        print(f"  n_genes={g}...")
        t = benchmark_single(grid_side=baseline["grid_side"], n_genes=g, K=baseline["K"], D=baseline["D"], iters=baseline["iters"])
        times_genes.append(t)

    print("\n=== 3/4: Benchmarking K sweep ===")
    times_k = []
    for k in sweep_k:
        print(f"  K={k}...")
        t = benchmark_single(grid_side=baseline["grid_side"], n_genes=baseline["n_genes"], K=k, D=baseline["D"], iters=baseline["iters"])
        times_k.append(t)

    print("\n=== 4/4: Benchmarking D sweep ===")
    times_d = []
    for d in sweep_d:
        print(f"  D={d}...")
        t = benchmark_single(grid_side=baseline["grid_side"], n_genes=baseline["n_genes"], K=baseline["K"], D=d, iters=baseline["iters"])
        times_d.append(t)

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    def style_ax(ax, title, xlabel, ylabel="Time per Iteration [s] (excl. Iter 1)", show_legend=False):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        if show_legend:
            ax.legend(frameon=True)

    # 1. grid_side (Quadratic reference: O(N^2)) retained
    base_idx = sweep_grid.index(baseline["grid_side"])
    ref_scale_gs = times_grid[base_idx] / (baseline["grid_side"] ** 2)
    grid_dense = np.linspace(sweep_grid[0], sweep_grid[-1], 200)

    axes[0].plot(sweep_grid, times_grid, "o-", color="#1f77b4", label="Observed", lw=2)
    axes[0].plot(grid_dense, ref_scale_gs * (grid_dense ** 2), "--", color="crimson", label=r"Ref: Quadratic $O(\mathrm{grid}^2)$")
    style_ax(axes[0], "Scaling vs. Grid Side Length", "grid_side", show_legend=True)

    # 2. grid_side * grid_side (Observed only)
    spots = [gs * gs for gs in sweep_grid]
    axes[1].plot(spots, times_grid, "s-", color="#2ca02c", lw=2)
    style_ax(axes[1], "Scaling vs. Total Spots (grid_side²)", "n_spots = grid_side²")

    # 3. n_genes (Observed only)
    axes[2].plot(sweep_genes, times_genes, "o-", color="#ff7f0e", lw=2)
    style_ax(axes[2], "Scaling vs. Number of Genes", "n_genes")

    # 4. K factors (Observed only)
    axes[3].plot(sweep_k, times_k, "o-", color="#9467bd", lw=2)
    style_ax(axes[3], "Scaling vs. Latent Factors (K)", "K (n_factors)")

    # 5. D length scales (Observed only)
    axes[4].plot(sweep_d, times_d, "o-", color="#8c564b", lw=2)
    style_ax(axes[4], "Scaling vs. Length Scales (D)", "D (n_length_scales)")

    # Hide unused 6th subplot
    axes[5].axis("off")

    fig.suptitle("WaveFactor Computational Scaling per Iteration (Excl. Iteration 1)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join(SCRIPT_DIR, "wavefactor_scaling_benchmarks.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved plots to {output_path}")


if __name__ == "__main__":
    run_all_benchmarks()