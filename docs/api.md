# API Reference

This page contains the automated API reference for WaveFactor. Signatures, type annotations, parameters, and return types are parsed directly from the package docstrings.

---

## Estimator

::: wavefactor.WaveFactor
    options:
      members:
        - fit
        - transform
        - fit_transform
        - get_result
        - factors_
        - loadings_
        - elbo_
        - elbo_history_
        - spatial_pip_
        - gene_pip_
        - precision_t_
        - precision_tau_

---

## Results Container

::: wavefactor.results.WaveFactorResult
    options:
      members:
        - factors
        - spatial_factor_maps
        - loadings
        - gene_pip
        - spatial_pip
        - precision_t
        - precision_tau
        - elbo
        - elbo_history
        - cpp_time
        - dimensions
        - grid_side_length
        - grid_shape
        - summary

---

## Spatial Preprocessing

::: wavefactor.data.prepare_spatial_data
