"""
Integration tests for WaveFactor estimator and WaveFactorResult.
"""

from tests.compat import pytest
import numpy as np
from wavefactor import WaveFactor, WaveFactorResult
from wavefactor.engine import init_parameters


def test_wavefactor_estimator_fit_transform():
    np.random.seed(123)
    # Generate synthetic spatial data on 8x8 = 64 spots
    N_spots = 64
    N_genes = 6
    K = 2
    D = 2  # R = 3

    # Generate 2 spatial patterns on 8x8 integer grid
    gx = np.arange(8)
    gy = np.arange(8)
    xx, yy = np.meshgrid(gx, gy)
    pattern1 = np.sin(2 * np.pi * (xx / 8.0)).ravel()
    pattern2 = np.cos(2 * np.pi * (yy / 8.0)).ravel()

    true_S = np.column_stack([pattern1, pattern2])  # (64, 2)
    true_F = np.random.randn(K, N_genes)  # (2, 6)

    noise = np.random.randn(N_spots, N_genes) * 0.1
    expr = np.dot(true_S, true_F) + noise

    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Fit WaveFactor (defaults to init="prior")
    model = WaveFactor(
        n_factors=K,
        n_length_scales=D,
        spatial_prior=np.array([0.9, 0.5, 0.2]),
        gene_prior=np.full((K, N_genes), 0.5),
        max_iter=15,
        tol=1e-5,
        n_init=2,
        n_jobs=1,
        random_state=42,
        verbose=False,
    )

    factors = model.fit_transform(expr, coords)

    assert factors.shape == (N_spots, K)
    assert np.array_equal(factors, model.transform())
    assert model.factors_.shape == (N_spots, K)
    assert model.loadings_.shape == (K, N_genes)
    assert len(model.elbo_history_) > 0
    assert np.isfinite(model.elbo_)
    assert model.init == "prior"

    # Test result object
    res = model.get_result()
    assert isinstance(res, WaveFactorResult)
    assert res.n_factors == K
    assert res.n_features == N_genes
    assert res.grid_side_length == 8
    assert res.grid_shape == (8, 8)
    assert res.spatial_factor_maps.shape == (8, 8, K)
    assert res.spot_grid_coords.shape == (N_spots, 2)
    assert np.array_equal(res.grid_x, res.spot_grid_coords[:, 0])
    assert np.array_equal(res.grid_y, res.spot_grid_coords[:, 1])
    assert res.gene_pip.shape == (K, N_genes)
    assert np.array_equal(model.gene_pip_, res.gene_pip)
    assert len(res.spatial_pip) == K
    assert len(model.spatial_pip_) == K
    # Test full variational distribution properties
    assert isinstance(res.mu_L, list)
    assert isinstance(res.sigma_squared_L, list)
    assert res.mu_F.shape == (K, N_genes)
    assert res.sigma_squared_F.shape == (K, N_genes)
    assert res.loadings_variance.shape == (K, N_genes)
    assert res.alpha_hat_t.shape == (D + 1, K)
    assert res.beta_hat_t.shape == (D + 1, K)
    assert res.alpha_hat_tau.shape == (D + 1, N_genes)
    assert res.beta_hat_tau.shape == (D + 1, N_genes)
    assert np.all(res.loadings_variance >= 0.0)

    # Test provenance and rich string representations
    assert model.version_ == res.version
    assert isinstance(model.version_, str)
    assert len(model.version_) > 0
    assert model.fit_info_["version"] == res.version
    assert model.fit_info_["n_factors"] == K
    assert model.fit_info_["n_spots"] == N_spots
    assert model.fit_info_["n_features"] == N_genes
    assert model.fit_info_["n_init"] == 2
    assert res.n_spots == N_spots
    assert res.n_init == 2
    assert "version=v" in repr(res)
    assert f"{K} factors" in repr(res)
    assert "WaveFactor Analysis Result" in res.summary()
    assert f"Version:             {res.version}" in res.summary()
    assert "Features:            6" in res.summary()
    assert "Initialisations:     Best of 2 starts" in res.summary()
    assert "C++ Engine Runtime" not in res.summary()

    # Test RuntimeError on unfitted model
    unfitted = WaveFactor()
    with pytest.raises(RuntimeError):
        unfitted.transform()


def test_estimator_get_set_params():
    model = WaveFactor(n_factors=5, n_length_scales=3, max_iter=200)
    params = model.get_params()
    assert params["n_factors"] == 5
    assert params["n_length_scales"] == 3
    assert params["max_iter"] == 200
    assert params["init"] == "prior"

    model.set_params(n_factors=8, max_iter=500, init="random")
    assert model.n_factors == 8
    assert model.max_iter == 500
    assert model.init == "random"


def test_estimator_invalid_init():
    """Asserts ValueError is raised on unknown init mode."""
    with pytest.raises(ValueError, match="Unknown init mode 'invalid'"):
        init_parameters(
            Y=[],
            dimensions={
                "n_resolutions": 2,
                "n_factors": 2,
                "F_shape": (2, 4),
                "ab_t_shape": (2, 2),
                "ab_tau_shape": (2, 4),
                "L_shape": [[[]]],
            },
            init="invalid",
        )

    model = WaveFactor(n_length_scales=1, init="unsupported", verbose=False)
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expr = np.ones((4, 2))
    with pytest.raises(ValueError, match="Unknown init mode 'unsupported'"):
        model.fit(expr, coords)


def test_version_provenance():
    """Asserts wavefactor.__version__ resolves correctly from repository root asset."""
    import wavefactor
    from pathlib import Path

    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    assert version_file.is_file()
    expected_version = version_file.read_text(encoding="utf-8").strip()

    assert wavefactor.__version__ == expected_version
    assert isinstance(wavefactor.__version__, str)
    assert len(wavefactor.__version__.split(".")) >= 2


def test_console_banner_verbal_emission():
    """Asserts verbose fit() emits verbal disambiguated banner without pronumerals."""
    import io
    from contextlib import redirect_stdout
    import wavefactor

    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expr = np.ones((4, 2))

    model = WaveFactor(n_factors=2, n_length_scales=1, max_iter=2, verbose=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        model.fit(expr, coords)

    output = buf.getvalue()
    assert f"[WaveFactor v{wavefactor.__version__}]" in output
    assert "Fitting 2 factors (1 detail scales, 2 coefficient resolutions) on 4 spots x 2 features..." in output


