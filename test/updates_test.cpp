#include <gtest/gtest.h>
#include "updates.hpp"
#include "mocks.hpp"
#include <unsupported/Eigen/SpecialFunctions>

TEST(CaviUpdatesTest, ComputeUpdateSigmaSquaredL)
{
    EXPECT_NEAR(compute_update_sigma_squared_L(1, 1, 1, 1, mocks::parameters), mocks::update_sigma_squared_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, MakeLZLUpdateContextForResolution)
{
    // Uniform mocks: shapes, zero diagonal and no remaining NaNs
    for (int i = 0; i < mocks::parameters.n_resolutions; ++i)
    {
        LZLUpdateContextForResolution ctx_resolution_i = make_L_Z_L_update_context_for_resolution(i, mocks::parameters);
        EXPECT_EQ(ctx_resolution_i.phi_L_mat.rows(), mocks::parameters.n_factors);
        EXPECT_EQ(ctx_resolution_i.phi_L_mat.cols(), mocks::parameters.n_factors);
        EXPECT_EQ(ctx_resolution_i.dot_Y_mat.rows(), mocks::parameters.wavelet_indices[i].n_coefficients);
        EXPECT_EQ(ctx_resolution_i.dot_Y_mat.cols(), mocks::parameters.n_factors);
        EXPECT_FALSE(ctx_resolution_i.phi_L_mat.hasNaN());
        EXPECT_FALSE(ctx_resolution_i.dot_Y_mat.hasNaN());
        for (int l = 0; l < mocks::parameters.n_factors; ++l)
        {
            EXPECT_EQ(ctx_resolution_i.phi_L_mat(l, l), 0.0);
        }
    }

    // Asymmetric parameters (distinct mu_F per factor) so a wrong factor column cannot go unnoticed
    Parameters asymmetric_params = mocks::parameters;
    asymmetric_params.mu_F = {{1, 2, 3}, {4, 5, 6}};
    for (int i = 0; i < asymmetric_params.n_resolutions; ++i)
    {
        LZLUpdateContextForResolution ctx_resolution_i = make_L_Z_L_update_context_for_resolution(i, asymmetric_params);
        for (int l = 0; l < asymmetric_params.n_factors; ++l)
        {
            for (int m = 0; m < asymmetric_params.n_factors; ++m)
            {
                double expected_phi_L_m_l = 0.0;
                if (m != l)
                {
                    for (int d = 0; d < asymmetric_params.n_features; ++d)
                    {
                        expected_phi_L_m_l += gamma_tau(i, d, asymmetric_params) * xi_F(l, d, asymmetric_params) * xi_F(m, d, asymmetric_params);
                    }
                }
                EXPECT_NEAR(ctx_resolution_i.phi_L_mat(m, l), expected_phi_L_m_l, 1e-12);
            }
            for (int p = 0; p < asymmetric_params.wavelet_indices[i].n_coefficients; ++p)
            {
                double expected_dot_Y_p_l = 0.0;
                for (int d = 0; d < asymmetric_params.n_features; ++d)
                {
                    expected_dot_Y_p_l += asymmetric_params.Y_mats[i](p, d) * gamma_tau(i, d, asymmetric_params) * xi_F(l, d, asymmetric_params);
                }
                EXPECT_NEAR(ctx_resolution_i.dot_Y_mat(p, l), expected_dot_Y_p_l, 1e-12);
            }
        }
    }
}

TEST(CaviUpdatesTest, MakeLZLUpdateContextForResolutionFactor)
{
    LZLUpdateContextForResolution ctx_resolution_1 = make_L_Z_L_update_context_for_resolution(1, mocks::parameters);
    LZLUpdateContextForResolutionFactor ctx_resolution_factor_i_l = make_L_Z_L_update_context_for_resolution_factor(1, 1, ctx_resolution_1, mocks::parameters);
    EXPECT_NEAR(ctx_resolution_factor_i_l.update_sigma_squared_L, mocks::update_sigma_squared_L_ijk_l, 0.001);
    EXPECT_EQ(ctx_resolution_factor_i_l.phi_L.size(), static_cast<std::size_t>(mocks::parameters.n_factors));
    // nu_L = gamma_tau * xi_F = 0.5 per feature, so dot_Y = 3 * 0.5 and phi_L[m != l] = 3 * 0.5 * 0.5
    for (std::size_t j = 0; j < ctx_resolution_factor_i_l.dot_Y.size(); ++j)
    {
        for (std::size_t k = 0; k < ctx_resolution_factor_i_l.dot_Y[j].size(); ++k)
        {
            EXPECT_NEAR(ctx_resolution_factor_i_l.dot_Y[j][k], 1.5, 0.001);
        }
    }
    EXPECT_NEAR(ctx_resolution_factor_i_l.phi_L[0], 0.75, 0.001);
    // Verify explicit self-interaction zeroing and no remaining NaNs
    EXPECT_EQ(ctx_resolution_factor_i_l.phi_L[1], 0.0);
    for (int m = 0; m < mocks::parameters.n_factors; ++m)
    {
        EXPECT_FALSE(std::isnan(ctx_resolution_factor_i_l.phi_L[m]));
    }

    // Verify column slicing for every factor on asymmetric parameters
    Parameters asymmetric_params = mocks::parameters;
    asymmetric_params.mu_F = {{1, 2, 3}, {4, 5, 6}};
    LZLUpdateContextForResolution ctx_resolution_asym = make_L_Z_L_update_context_for_resolution(1, asymmetric_params);
    for (int l = 0; l < asymmetric_params.n_factors; ++l)
    {
        LZLUpdateContextForResolutionFactor ctx_asym_resolution_factor_l = make_L_Z_L_update_context_for_resolution_factor(1, l, ctx_resolution_asym, asymmetric_params);
        for (int m = 0; m < asymmetric_params.n_factors; ++m)
        {
            EXPECT_DOUBLE_EQ(ctx_asym_resolution_factor_l.phi_L[m], ctx_resolution_asym.phi_L_mat(m, l));
        }
        for (std::size_t j = 0; j < ctx_asym_resolution_factor_l.dot_Y.size(); ++j)
        {
            for (std::size_t k = 0; k < ctx_asym_resolution_factor_l.dot_Y[j].size(); ++k)
            {
                int p = asymmetric_params.wavelet_indices[1].subband_to_flat_index[j][k];
                EXPECT_DOUBLE_EQ(ctx_asym_resolution_factor_l.dot_Y[j][k], ctx_resolution_asym.dot_Y_mat(p, l));
            }
        }
    }
}

TEST(CaviUpdatesTest, ComputeUpdateLZL)
{
    LZLUpdateContextForResolution ctx_resolution_1 = make_L_Z_L_update_context_for_resolution(1, mocks::parameters);
    LZLUpdateContextForResolutionFactor ctx_resolution_factor_i_l = make_L_Z_L_update_context_for_resolution_factor(1, 1, ctx_resolution_1, mocks::parameters);
    auto update_L_Z_L = compute_update_L_Z_L(1, 1, 1, 1, ctx_resolution_factor_i_l, mocks::parameters);
    EXPECT_NEAR(update_L_Z_L.update_sigma_squared_L, mocks::update_sigma_squared_L_ijk_l, 0.001);
    EXPECT_NEAR(update_L_Z_L.update_mu_L, mocks::update_mu_L_ijk_l, 0.001);
    EXPECT_NEAR(update_L_Z_L.update_log_r_L, mocks::update_log_r_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, MakeFZFTauUpdateContext)
{
    FZFTauUpdateContext ctx = make_F_Z_F_tau_update_context(mocks::parameters);
    EXPECT_EQ(ctx.lambda_bar_L.size(), static_cast<std::size_t>(mocks::parameters.n_resolutions));
    EXPECT_EQ(ctx.phi_F.size(), static_cast<std::size_t>(mocks::parameters.n_resolutions));
    EXPECT_EQ(ctx.nu_F.size(), static_cast<std::size_t>(mocks::parameters.n_resolutions));

    for (int a = 0; a < mocks::parameters.n_resolutions; ++a)
    {
        EXPECT_EQ(ctx.lambda_bar_L[a].size(), static_cast<std::size_t>(mocks::parameters.n_factors));
        EXPECT_EQ(ctx.phi_F[a].size(), static_cast<std::size_t>(mocks::parameters.n_factors));
        EXPECT_EQ(ctx.nu_F[a].size(), static_cast<std::size_t>(mocks::parameters.n_factors));

        for (int m = 0; m < mocks::parameters.n_factors; ++m)
        {
            EXPECT_FALSE(std::isnan(ctx.lambda_bar_L[a][m]));
            EXPECT_EQ(ctx.phi_F[a][m].size(), static_cast<std::size_t>(mocks::parameters.n_factors));
            EXPECT_EQ(ctx.nu_F[a][m].size(), static_cast<std::size_t>(mocks::parameters.n_features));

            // Verify diagonal zeroing and symmetry of Gram matrix
            EXPECT_EQ(ctx.phi_F[a][m][m], 0.0);
            for (int mp = 0; mp < mocks::parameters.n_factors; ++mp)
            {
                EXPECT_FALSE(std::isnan(ctx.phi_F[a][m][mp]));
                EXPECT_DOUBLE_EQ(ctx.phi_F[a][m][mp], ctx.phi_F[a][mp][m]);
            }
            for (int g = 0; g < mocks::parameters.n_features; ++g)
            {
                EXPECT_FALSE(std::isnan(ctx.nu_F[a][m][g]));
            }
        }
    }
}

TEST(CaviUpdatesTest, MakeFZFUpdateContextForFactor)
{
    FZFTauUpdateContext ctx_F_Z_F_tau = make_F_Z_F_tau_update_context(mocks::parameters);
    FZFUpdateContextForFactor ctx_factor_1 = make_F_Z_F_update_context_for_factor(1, ctx_F_Z_F_tau, mocks::parameters);
    EXPECT_NEAR(ctx_factor_1.s_bar_L[1], mocks::s_bar_L_i_j, 0.001);
    EXPECT_NEAR(ctx_factor_1.sigma_squared_F[1], mocks::update_sigma_squared_F_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateFZF)
{
    FZFTauUpdateContext ctx_F_Z_F_tau = make_F_Z_F_tau_update_context(mocks::parameters);
    FZFUpdateContextForFactor ctx_factor_1 = make_F_Z_F_update_context_for_factor(1, ctx_F_Z_F_tau, mocks::parameters);
    auto update_F_Z_F = compute_update_F_Z_F(1, 1, ctx_factor_1, mocks::parameters);
    EXPECT_NEAR(update_F_Z_F.update_sigma_squared_F, mocks::update_sigma_squared_F_i_j, 0.001);
    EXPECT_NEAR(update_F_Z_F.update_mu_F, mocks::update_mu_F_i_j, 0.001);
    EXPECT_NEAR(update_F_Z_F.update_log_r_F, mocks::update_log_r_F_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateAlphaHatTau)
{
    EXPECT_NEAR(compute_update_alpha_hat_tau(0, 1, mocks::parameters), mocks::update_alpha_hat_tau_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_alpha_hat_tau(1, 1, mocks::parameters), mocks::update_alpha_hat_tau_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateTau)
{
    FZFTauUpdateContext ctx = make_F_Z_F_tau_update_context(mocks::parameters);
    EXPECT_NEAR(compute_update_beta_hat_tau(0, 1, ctx, mocks::parameters), mocks::update_beta_hat_tau_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_beta_hat_tau(1, 1, ctx, mocks::parameters), mocks::update_beta_hat_tau_i_l_1, 0.001);

    auto update_tau = compute_update_tau(1, 1, ctx, mocks::parameters);
    EXPECT_NEAR(update_tau.update_alpha_hat_tau, mocks::update_alpha_hat_tau_i_l_1, 0.001);
    EXPECT_NEAR(update_tau.update_beta_hat_tau, mocks::update_beta_hat_tau_i_l_1, 0.001);

    // Verify defensive NaN poisoning: passing signaling NaN for sum_Y_sq must propagate NaN
    Parameters poisoned_params = mocks::parameters;
    poisoned_params.sum_Y_sq[1][1] = std::numeric_limits<double>::signaling_NaN();
    EXPECT_TRUE(std::isnan(compute_update_beta_hat_tau(1, 1, ctx, poisoned_params)));
}

TEST(CaviUpdatesTest, ComputeUpdateAlphaHatT)
{
    EXPECT_NEAR(compute_update_alpha_hat_t(0, 1, mocks::parameters), mocks::update_alpha_hat_t_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_alpha_hat_t(1, 1, mocks::parameters), mocks::update_alpha_hat_t_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateBetaHatT)
{
    EXPECT_NEAR(compute_update_beta_hat_t(0, 1, mocks::parameters), mocks::update_beta_hat_t_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_beta_hat_t(1, 1, mocks::parameters), mocks::update_beta_hat_t_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateT)
{
    auto update_t = compute_update_t(1, 1, mocks::parameters);
    EXPECT_NEAR(update_t.update_alpha_hat_t, mocks::update_alpha_hat_t_i_l_1, 0.001);
    EXPECT_NEAR(update_t.update_beta_hat_t, mocks::update_beta_hat_t_i_l_1, 0.001);
}