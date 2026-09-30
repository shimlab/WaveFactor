#include <gtest/gtest.h>
#include "elbo.hpp"
#include "mocks.hpp"
#include <cmath>

TEST(CaviElboTest, ComputeELogLikelihoodYijkLGivenPiLFtau)
{
    EXPECT_NEAR(compute_E_log_likelihood_Y_ijk_l_given_pi_L_F_tau(1, 1, 1, 1, mocks::parameters), mocks::E_log_likelihood_Y_ijk_l_given_pi_L_F_tau, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodLijklGivenZLt)
{
    EXPECT_NEAR(compute_E_log_likelihood_L_ijk_l_given_Z_L_t(1, 1, 1, 1, mocks::parameters), mocks::E_log_likelihood_L_ijk_l_given_Z_L_t, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodFijGivenZF)
{
    EXPECT_NEAR(compute_E_log_likelihood_F_i_j_given_Z_F(1, 1, mocks::parameters), mocks::E_log_likelihood_F_i_j_given_Z_F, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodZLijkl)
{
    EXPECT_NEAR(compute_E_log_likelihood_Z_L_ijk_l(1, 1, 1, 1, mocks::parameters), mocks::E_log_likelihood_Z_L_ijk_l, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodZFij)
{
    EXPECT_NEAR(compute_E_log_likelihood_Z_F_i_j(1, 1, mocks::parameters), mocks::E_log_likelihood_Z_F_i_j, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodTil)
{
    EXPECT_NEAR(compute_E_log_likelihood_t_i_l(1, 1, mocks::parameters), mocks::E_log_likelihood_t_i_l, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodTauil)
{
    EXPECT_NEAR(compute_E_log_likelihood_tau_i_l(1, 1, mocks::parameters), mocks::E_log_likelihood_tau_i_l, 0.001);
}

TEST(CaviElboTest, ComputeENegativeVariationalLogLikelihoodLijklZLikjl)
{
    EXPECT_NEAR(compute_E_negative_variational_log_likelihood_L_ijk_l_Z_L_ijk_l(1, 1, 1, 1, mocks::parameters), mocks::E_negative_variational_log_likelihood_L_ijk_l_Z_L_ijk_l, 0.001);
}

TEST(CaviElboTest, ComputeENegativeVariationalLogLikelihoodFijZFij)
{
    EXPECT_NEAR(compute_E_negative_variational_log_likelihood_F_i_j_Z_F_i_j(1, 1, mocks::parameters), mocks::E_negative_variational_log_likelihood_F_i_j_Z_F_i_j, 0.001);
}

TEST(CaviElboTest, ComputeENegativeVariationalLogLikelihoodTil)
{
    EXPECT_NEAR(compute_E_negative_variational_log_likelihood_t_i_l(1, 1, mocks::parameters), mocks::E_negative_variational_log_likelihood_t_i_l, 0.001);
}

TEST(CaviElboTest, ComputeENegativeVariationalLogLikelihoodTauil)
{
    EXPECT_NEAR(compute_E_negative_variational_log_likelihood_tau_i_l(1, 1, mocks::parameters), mocks::E_negative_variational_log_likelihood_tau_i_l, 0.001);
}

TEST(CaviElboTest, ComputeElbo)
{
    EXPECT_NEAR(compute_elbo(mocks::parameters), mocks::elbo, 0.001);
}