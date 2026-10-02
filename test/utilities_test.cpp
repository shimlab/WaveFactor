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
    LZLUpdateContext ctx_i_l = make_L_Z_L_update_context(1, 1, mocks::parameters);
    EXPECT_NEAR(s_bar_F(1, 1, 1, 1, ctx_i_l, mocks::parameters), mocks::s_bar_F_ijk_l, 0.001);
}

TEST(UtilitiesTest, UBarF)
{
    EXPECT_NEAR(u_bar_F(1, 1, mocks::parameters), mocks::u_bar_F_i_l, 0.001);
}

TEST(UtilitiesTest, UL)
{
    EXPECT_NEAR(u_L(1, 1, 1, 1, 1, mocks::parameters), mocks::u_L_abc_i_j, 0.001);
}

TEST(UtilitiesTest, SBarLWithContext)
{
    FZFUpdateContext ctx_i = make_F_Z_F_update_context(1, mocks::parameters);
    EXPECT_NEAR(s_bar_L(1, 1, ctx_i, mocks::parameters), mocks::s_bar_L_i_j, 0.001);
}

TEST(UtilitiesTest, UBarL)
{
    EXPECT_NEAR(u_bar_L(1, 1, mocks::parameters), mocks::u_bar_L_i_j, 0.001);
}