#include <gtest/gtest.h>
#include "utilities.hpp"
#include "updates.hpp"
#include "mocks.hpp"
#include <unsupported/Eigen/SpecialFunctions>
#include <cmath>

TEST(UtilitiesTest, SumLog)
{
    const double log_a = -4.60517;
    const double log_b = -6.90776;
    EXPECT_NEAR(sum_log(log_a, log_b), -4.50986, 0.001);
}

TEST(UtilitiesTest, GammaT)
{
    EXPECT_NEAR(gamma_t(1, 1, mocks::parameters), mocks::gamma_t_i_l, 0.001);
}

TEST(UtilitiesTest, GammaTau)
{
    EXPECT_NEAR(gamma_tau(1, 1, mocks::parameters), mocks::gamma_tau_i_l, 0.001);
}

TEST(UtilitiesTest, XiL)
{
    EXPECT_NEAR(xi_L(1, 1, 1, 1, mocks::parameters), mocks::xi_L_ijk_l, 0.001);
}

TEST(UtilitiesTest, XiF)
{
    EXPECT_NEAR(xi_F(1, 1, mocks::parameters), mocks::xi_F_i_j, 0.001);
}

TEST(UtilitiesTest, LambdaL)
{
    EXPECT_NEAR(lambda_L(1, 1, 1, 1, mocks::parameters), mocks::lambda_L_ijk_l, 0.001);
}

TEST(UtilitiesTest, LambdaF)
{
    EXPECT_NEAR(lambda_F(1, 1, mocks::parameters), mocks::lambda_F_i_j, 0.001);
}

TEST(UtilitiesTest, ThetaT)
{
    EXPECT_NEAR(theta_t(1, 1, mocks::parameters), mocks::theta_t_i_l, 0.001);
}

TEST(UtilitiesTest, ThetaTau)
{
    EXPECT_NEAR(theta_tau(1, 1, mocks::parameters), mocks::theta_tau_i_l, 0.001);
}

TEST(UtilitiesTest, UF)
{
    EXPECT_NEAR(u_F(1, 1, 1, mocks::parameters), mocks::u_F_i_l_d, 0.001);
}

TEST(UtilitiesTest, SBarFWithContext)
{
    LZLUpdateContextForResolution ctx_resolution_1 = make_L_Z_L_update_context_for_resolution(1, mocks::parameters);
    LZLUpdateContextForResolutionFactor ctx_resolution_factor_i_l = make_L_Z_L_update_context_for_resolution_factor(1, 1, ctx_resolution_1, mocks::parameters);
    EXPECT_NEAR(s_bar_F(1, 1, 1, 1, ctx_resolution_factor_i_l, mocks::parameters), mocks::s_bar_F_ijk_l, 0.001);
}

TEST(UtilitiesTest, UBarF)
{
    EXPECT_NEAR(u_bar_F(1, 1, mocks::parameters), mocks::u_bar_F_i_l, 0.001);
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