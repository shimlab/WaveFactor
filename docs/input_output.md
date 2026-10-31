# Data Input & Output

## 1. What WaveFactor Expects as Input

WaveFactor needs two inputs: an expression matrix `X` and spot coordinates `coords`.

### A. The Spatial Grid

- **Square Grid**: The data must sit on a square grid where each side length $L$ is a power of 2 (e.g., $16 \times 16$, $32 \times 32$, or $64 \times 64$).
- **Total Spots**: The total number of spots must be $L^2$ (a power of 4, such as 256, 1024, or 4096).
- **Coordinates (`coords`)**: A 2D array of shape `(N_spots, 2)` with integers from $0$ to $L-1$ as ordinates. Each grid location must appear exactly once.

### B. Expression Matrix (`X`)
- A 2D array of shape `(N_spots, N_genes)`.

---

## 2. Key Parameters

- `n_factors`: Number of latent spatial patterns to find (e.g. $5$ or $10$).
- `n_length_scales`: Number of wavelet detail levels ($D$). Must satisfy $2^D \le L$.
- `n_init`: Number of random initializations (WaveFactor keeps the best run based on ELBO).

---

## 3. What WaveFactor Returns (`result`)

After fitting the model with `model.fit(X, coords)`, access the posterior estimates via `result = model.get_result()`:

```python
model.fit(X, coords)
result = model.get_result()
```

### Key Output Fields in `result`

| Output | Shape | Meaning |
| :--- | :--- | :--- |
| `result.factors` | `(N_spots, K)` | Spatial factor values at each spot. |
| `result.spatial_factor_maps` | `(L, L, K)` | Reconstructed 2D spatial maps of each factor. |
| `result.loadings` | `(K, N_genes)` | How strongly each gene belongs to each factor. |
| `result.gene_pip` | `(K, N_genes)` | Gene inclusion probability ($\approx 1$ = active gene, $\approx 0$ = noise). |
| `result.spatial_pip` | List of arrays | Wavelet inclusion probabilities across spatial length scales. |
| `result.elbo` | `float` | Final model fit score (higher is better). |

---

!!! tip "Full API Reference"
    For full specification of parameters and fields, please visit the [API Reference](api.md).
