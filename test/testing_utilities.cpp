#include "testing_utilities.hpp"

// Helper function to compare two std::vector<double> using relative tolerance
void compareVectorDouble(const std::vector<double> &v1, const std::vector<double> &v2, double rtol, double atol)
{
    ASSERT_EQ(v1.size(), v2.size());
    for (size_t i = 0; i < v1.size(); ++i)
    {
        EXPECT_TRUE(testing_utilities::isCloseRel(v1[i], v2[i], rtol, atol))
            << "Element mismatch at index " << i << ": " << v1[i] << " vs " << v2[i];
    }
}

// Helper function to compare two 1D tensors using relative tolerance
void compareTensor1D(const Tensor1D &t1, const Tensor1D &t2, double rtol, double atol)
{
    ASSERT_EQ(t1.size(), t2.size());
    for (size_t i = 0; i < t1.size(); ++i)
    {
        EXPECT_TRUE(testing_utilities::isCloseRel(t1[i], t2[i], rtol, atol))
            << "Element mismatch at index " << i << ": " << t1[i] << " vs " << t2[i];
    }
}

// Helper function to compare two 2D tensors using relative tolerance
void compareTensor2D(const Tensor2D &t1, const Tensor2D &t2, double rtol, double atol)
{
    ASSERT_EQ(t1.size(), t2.size());
    for (size_t i = 0; i < t1.size(); ++i)
    {
        compareTensor1D(t1[i], t2[i], rtol, atol);
    }
}

// Helper function to compare two 3D tensors using relative tolerance
void compareTensor3D(const Tensor3D &t1, const Tensor3D &t2, double rtol, double atol)
{
    ASSERT_EQ(t1.size(), t2.size());
    for (size_t i = 0; i < t1.size(); ++i)
    {
        compareTensor2D(t1[i], t2[i], rtol, atol);
    }
}

// Helper function to compare two 4D tensors using relative tolerance
void compareTensor4D(const Tensor4D &t1, const Tensor4D &t2, double rtol, double atol)
{
    ASSERT_EQ(t1.size(), t2.size());
    for (size_t i = 0; i < t1.size(); ++i)
    {
        compareTensor3D(t1[i], t2[i], rtol, atol);
    }
}

// Implementation of the VectorException
VectorException::VectorException(const std::vector<double> &vec)
    : std::runtime_error(formatMessage(vec)), vector_(vec) {}

const std::vector<double> &VectorException::getVector() const
{
    return vector_;
}

std::string VectorException::formatMessage(const std::vector<double> &vec)
{
    std::ostringstream oss;
    oss << "Vector values: ";
    for (const auto &val : vec)
    {
        oss << val << " ";
    }
    return oss.str();
}

void throwVectorException(const std::vector<double> &vec)
{
    throw VectorException(vec);
}