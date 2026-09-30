#include <gtest/gtest.h>
#include "updates.hpp"
#include "mocks.hpp"
#include <unsupported/Eigen/SpecialFunctions>

TEST(CaviUpdatesTest, ComputeUpdateSigmaSquaredL)
{
    EXPECT_NEAR(compute_update_sigma_squared_L(1, 1, 1, 1, mocks::parameters), mocks::update_sigma_squared_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateMuL)
{
    EXPECT_NEAR(compute_update_mu_L(1, 1, 1, 1, mocks::update_sigma_squared_L_ijk_l, mocks::parameters), mocks::update_mu_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateZLRelativePmf)
{
    EXPECT_NEAR(compute_update_Z_L_log_relative_pmf(1, 1, 1, 1, 0, mocks::update_sigma_squared_L_ijk_l, mocks::update_mu_L_ijk_l, mocks::parameters), mocks::update_Z_L_log_relative_pmf_0, 0.001);
    EXPECT_NEAR(compute_update_Z_L_log_relative_pmf(1, 1, 1, 1, 1, mocks::update_sigma_squared_L_ijk_l, mocks::update_mu_L_ijk_l, mocks::parameters), mocks::update_Z_L_log_relative_pmf_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateLogRL)
{
    EXPECT_NEAR(compute_update_log_r_L(1, 1, 1, 1, mocks::update_sigma_squared_L_ijk_l, mocks::update_mu_L_ijk_l, mocks::parameters), mocks::update_log_r_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateL)
{
    auto update_L = compute_update_L(1, 1, 1, 1, mocks::parameters);
    EXPECT_NEAR(update_L.sigma_squared_L, mocks::update_sigma_squared_L_ijk_l, 0.001);
    EXPECT_NEAR(update_L.mu_L, mocks::update_mu_L_ijk_l, 0.001);
    EXPECT_NEAR(update_L.log_r_L, mocks::update_log_r_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateSigmaSquaredF)
{
    EXPECT_NEAR(compute_update_sigma_squared_F(1, 1, mocks::parameters), mocks::update_sigma_squared_F_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateMuF)
{
    EXPECT_NEAR(compute_update_mu_F(1, 1, mocks::update_sigma_squared_F_i_j, mocks::parameters), mocks::update_mu_F_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateZFRelativePmf)
{
    EXPECT_NEAR(compute_update_Z_F_log_relative_pmf(1, 1, 0, mocks::update_sigma_squared_F_i_j, mocks::update_mu_F_i_j, mocks::parameters), mocks::update_Z_F_log_relative_pmf_0, 0.001);
    EXPECT_NEAR(compute_update_Z_F_log_relative_pmf(1, 1, 1, mocks::update_sigma_squared_F_i_j, mocks::update_mu_F_i_j, mocks::parameters), mocks::update_Z_F_log_relative_pmf_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateLogRF)
{
    EXPECT_NEAR(compute_update_log_r_F(1, 1, mocks::update_sigma_squared_F_i_j, mocks::update_mu_F_i_j, mocks::parameters), mocks::update_log_r_F_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateF)
{
    auto update_F = compute_update_F(1, 1, mocks::parameters);
    EXPECT_NEAR(update_F.sigma_squared_F, mocks::update_sigma_squared_F_i_j, 0.001);
    EXPECT_NEAR(update_F.mu_F, mocks::update_mu_F_i_j, 0.001);
    EXPECT_NEAR(update_F.log_r_F, mocks::update_log_r_F_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateAlphaHatTau)
{
    EXPECT_NEAR(compute_update_alpha_hat_tau(0, 1, mocks::parameters), mocks::update_alpha_hat_tau_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_alpha_hat_tau(1, 1, mocks::parameters), mocks::update_alpha_hat_tau_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateBetaHatTau)
{
    EXPECT_NEAR(compute_update_beta_hat_tau(0, 1, mocks::parameters), mocks::update_beta_hat_tau_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_beta_hat_tau(1, 1, mocks::parameters), mocks::update_beta_hat_tau_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateTau)
{
    auto update_tau = compute_update_tau(1, 1, mocks::parameters);
    EXPECT_NEAR(update_tau.alpha_hat_tau, mocks::update_alpha_hat_tau_i_l_1, 0.001);
    EXPECT_NEAR(update_tau.beta_hat_tau, mocks::update_beta_hat_tau_i_l_1, 0.001);
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
    EXPECT_NEAR(update_t.alpha_hat_t, mocks::update_alpha_hat_t_i_l_1, 0.001);
    EXPECT_NEAR(update_t.beta_hat_t, mocks::update_beta_hat_t_i_l_1, 0.001);
}