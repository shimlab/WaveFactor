#include <gtest/gtest.h>
#include "utilities.hpp"
#include "updates.hpp"
#include "mocks.hpp"
#include "testing_utilities.hpp"
#include <unsupported/Eigen/SpecialFunctions>
#include <cmath>

TEST(UtilitiesTest, SumLog)
{
    const double log_a = -4.605170185988092;
    const double log_b = -6.907755278982137;
    const double expected = std::log(std::exp(log_a) + std::exp(log_b));
    EXPECT_REL_EQ(sum_log(log_a, log_b), expected);
}

TEST(UtilitiesTest, GammaT)
{
    EXPECT_REL_EQ(gamma_t(1, 1, mocks::parameters), mocks::gamma_t_i_l);
}

TEST(UtilitiesTest, GammaTau)
{
    EXPECT_REL_EQ(gamma_tau(1, 1, mocks::parameters), mocks::gamma_tau_i_l);
}

TEST(UtilitiesTest, XiL)
{
    EXPECT_REL_EQ(xi_L(1, 1, 1, 1, mocks::parameters), mocks::xi_L_ijk_l);
}

TEST(UtilitiesTest, XiF)
{
    EXPECT_REL_EQ(xi_F(1, 1, mocks::parameters), mocks::xi_F_i_j);
}

TEST(UtilitiesTest, LambdaL)
{
    EXPECT_REL_EQ(lambda_L(1, 1, 1, 1, mocks::parameters), mocks::lambda_L_ijk_l);
}

TEST(UtilitiesTest, LambdaF)
{
    EXPECT_REL_EQ(lambda_F(1, 1, mocks::parameters), mocks::lambda_F_i_j);
}

TEST(UtilitiesTest, ThetaT)
{
    EXPECT_REL_EQ(theta_t(1, 1, mocks::parameters), mocks::theta_t_i_l);
}

TEST(UtilitiesTest, ThetaTau)
{
    EXPECT_REL_EQ(theta_tau(1, 1, mocks::parameters), mocks::theta_tau_i_l);
}

TEST(UtilitiesTest, UF)
{
    EXPECT_REL_EQ(u_F(1, 1, 1, mocks::parameters), mocks::u_F_i_l_d);
}

TEST(UtilitiesTest, SBarFWithContext)
{
    LZLUpdateContextForResolution ctx_resolution_1 = make_L_Z_L_update_context_for_resolution(1, mocks::parameters);
    LZLUpdateContextForResolutionFactor ctx_resolution_factor_i_l = make_L_Z_L_update_context_for_resolution_factor(1, 1, ctx_resolution_1, mocks::parameters);
    EXPECT_REL_EQ(s_bar_F(1, 1, 1, 1, ctx_resolution_factor_i_l, mocks::parameters), mocks::s_bar_F_ijk_l);
}

TEST(UtilitiesTest, UBarF)
{
    EXPECT_REL_EQ(u_bar_F(1, 1, mocks::parameters), mocks::u_bar_F_i_l);
}

TEST(UtilitiesTest, ComputeXiFMat)
{
    Parameters asymmetric_params = mocks::parameters;
    asymmetric_params.mu_F = {{1, 2, 3}, {4, 5, 6}};
    for (const Parameters *params : {&mocks::parameters, &asymmetric_params})
    {
        Eigen::MatrixXd Xi_F_mat = compute_Xi_F_mat(*params);
        EXPECT_EQ(Xi_F_mat.rows(), params->n_factors);
        EXPECT_EQ(Xi_F_mat.cols(), params->n_features);
        for (int m = 0; m < params->n_factors; ++m)
        {
            for (int g = 0; g < params->n_features; ++g)
            {
                EXPECT_DOUBLE_EQ(Xi_F_mat(m, g), xi_F(m, g, *params));
            }
        }
    }
}