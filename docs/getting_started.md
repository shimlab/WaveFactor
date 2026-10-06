# Getting Started

> **Note**: This documentation is a **work in progress** and subject to further improvements.

First, follow **[Installation Guide](installation.md)** to install WaveFactor into a virtual environment and activate said environment.

## Example Code Snippet

This example shows how WaveFactor can be applied on data and how posterior estimates can be extracted.

```python
import wavefactor as wf

# 1. Simulating input data
# Input requirements:
#   X: continuous expression matrix (N_spots x N_genes)
#   coords: (N_spots x 2) integer grid coordinates covering [0, L-1] x [0, L-1]
#   N_spots = L * L must be an exact power of 4 (e.g. 64, 256, 1024, 4096)
np.random.seed(42)
coords = np.mgrid[0:32, 0:32].reshape(2, -1).T
X = np.random.randn(1024, 200)

# 2. Instantiate WaveFactor estimator
model = wf.WaveFactor(
    n_factors=10,               # Number of latent factors (K)
    n_length_scales=4,          # Wavelet detail levels (R = 5 total resolutions)
    n_init=5,                   # Multi-start initializations (selects best ELBO)
    n_jobs=1,                   # Parallel workers across initializations
    random_state=42,
)

# 3. Apply WaveFactor to the data
factors = model.fit_transform(X, coords)

# 4. Access posterior estimates
result = model.get_result()
print(f"Final ELBO: {result.elbo:.2f} across {result.n_iter} iterations")

spot_factors = result.factors        # Latent spatial factors (N_spots x K)
gene_loadings = result.loadings      # Factor loadings matrix (K x N_genes)
gene_pips = result.gene_pip          # Gene Posterior Inclusion Probabilities
spatial_pips = result.spatial_pip    # Multiresolution spatial wavelet PIPs
```

## Another Example With Simulated Data Based On Gene Program Ground Truths

WaveFactor includes additionally ready-to-run simulation example with ground truth gene programs, fits the model, and checks the results against the ground truth gene programs.

This example can be ran with the following command.

```bash
python examples/getting_started/run_analysis.py
```