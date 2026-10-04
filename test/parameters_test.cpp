#include <gtest/gtest.h>
#include "parameters.hpp"
#include "mocks.hpp"

TEST(ParametersTest, PrecomputedResolutionDimensions)
{
    EXPECT_EQ(mocks::parameters.N_coefs_per_res.size(), 2);
    EXPECT_DOUBLE_EQ(mocks::parameters.N_coefs_per_res[0], 4.0);
    EXPECT_DOUBLE_EQ(mocks::parameters.N_coefs_per_res[1], 12.0);

    // Verify precomputed wavelet index maps, flat Y matrices, and static sum_Y_sq
    EXPECT_EQ(mocks::parameters.wavelet_indices.size(), 2);
    EXPECT_EQ(mocks::parameters.wavelet_indices[0].N_i, 4);
    EXPECT_EQ(mocks::parameters.wavelet_indices[1].N_i, 12);
    EXPECT_EQ(mocks::parameters.Y_mats.size(), 2);
    EXPECT_EQ(mocks::parameters.Y_mats[0].rows(), 4);
    EXPECT_EQ(mocks::parameters.Y_mats[0].cols(), 3);
    EXPECT_EQ(mocks::parameters.Y_mats[1].rows(), 12);
    EXPECT_EQ(mocks::parameters.Y_mats[1].cols(), 3);
    EXPECT_EQ(mocks::parameters.sum_Y_sq.size(), 2);
    EXPECT_DOUBLE_EQ(mocks::parameters.sum_Y_sq[0][0], 4.0);
    EXPECT_DOUBLE_EQ(mocks::parameters.sum_Y_sq[1][0], 12.0);

    // Verify copy constructor preserves all precomputed structures
    Parameters copied = mocks::parameters;
    EXPECT_EQ(copied.N_coefs_per_res.size(), 2);
    EXPECT_DOUBLE_EQ(copied.N_coefs_per_res[0], 4.0);
    EXPECT_DOUBLE_EQ(copied.N_coefs_per_res[1], 12.0);
    EXPECT_EQ(copied.wavelet_indices.size(), 2);
    EXPECT_EQ(copied.Y_mats.size(), 2);
    EXPECT_EQ(copied.sum_Y_sq.size(), 2);

    // Verify copy assignment preserves all precomputed structures
    Parameters assigned = mocks::parameters2;
    assigned = mocks::parameters;
    EXPECT_EQ(assigned.N_coefs_per_res.size(), 2);
    EXPECT_DOUBLE_EQ(assigned.N_coefs_per_res[0], 4.0);
    EXPECT_DOUBLE_EQ(assigned.N_coefs_per_res[1], 12.0);
    EXPECT_EQ(assigned.wavelet_indices.size(), 2);
    EXPECT_EQ(assigned.Y_mats.size(), 2);
    EXPECT_EQ(assigned.sum_Y_sq.size(), 2);
}