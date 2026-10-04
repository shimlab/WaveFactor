#include <gtest/gtest.h>
#include "updates.hpp"
#include "mocks.hpp"
#include <unsupported/Eigen/SpecialFunctions>

TEST(CaviUpdatesTest, ComputeUpdateSigmaSquaredL)
{
    EXPECT_NEAR(compute_update_sigma_squared_L(1, 1, 1, 1, mocks::parameters), mocks::update_sigma_squared_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, MakeLZLUpdateContext)
{
    LZLUpdateContext ctx_i_l = make_L_Z_L_update_context(1, 1, mocks::parameters);
    EXPECT_NEAR(ctx_i_l.update_sigma_squared_L, mocks::update_sigma_squared_L_ijk_l, 0.001);
    EXPECT_EQ(ctx_i_l.nu_L.size(), static_cast<std::size_t>(mocks::parameters.n_features));
    EXPECT_EQ(ctx_i_l.phi_L.size(), static_cast<std::size_t>(mocks::parameters.n_factors));
    for (int d = 0; d < mocks::parameters.n_features; ++d)
    {
        EXPECT_FALSE(std::isnan(ctx_i_l.nu_L[d]));
        EXPECT_NEAR(ctx_i_l.nu_L[d], 0.5, 0.001);
    }
    // Verify explicit self-interaction zeroing and no remaining NaNs
    EXPECT_EQ(ctx_i_l.phi_L[1], 0.0);
    for (int m = 0; m < mocks::parameters.n_factors; ++m)
    {
        EXPECT_FALSE(std::isnan(ctx_i_l.phi_L[m]));
    }
}

TEST(CaviUpdatesTest, ComputeUpdateLZL)
{
    LZLUpdateContext ctx_i_l = make_L_Z_L_update_context(1, 1, mocks::parameters);
    auto update_L_Z_L = compute_update_L_Z_L(1, 1, 1, 1, ctx_i_l, mocks::parameters);
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