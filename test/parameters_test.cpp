#include <gtest/gtest.h>
#include "parameters.hpp"
#include "mocks.hpp"

TEST(ParametersTest, PrecomputedResolutionDimensions)
{
    EXPECT_EQ(mocks::parameters.N_coefs_per_res.size(), 2);
    EXPECT_DOUBLE_EQ(mocks::parameters.N_coefs_per_res[0], 4.0);
    EXPECT_DOUBLE_EQ(mocks::parameters.N_coefs_per_res[1], 12.0);

    // Verify copy constructor preserves N_coefs_per_res
    Parameters copied = mocks::parameters;
    EXPECT_EQ(copied.N_coefs_per_res.size(), 2);
    EXPECT_DOUBLE_EQ(copied.N_coefs_per_res[0], 4.0);
    EXPECT_DOUBLE_EQ(copied.N_coefs_per_res[1], 12.0);

    // Verify copy assignment preserves N_coefs_per_res
    Parameters assigned = mocks::parameters2;
    assigned = mocks::parameters;
    EXPECT_EQ(assigned.N_coefs_per_res.size(), 2);
    EXPECT_DOUBLE_EQ(assigned.N_coefs_per_res[0], 4.0);
    EXPECT_DOUBLE_EQ(assigned.N_coefs_per_res[1], 12.0);
}