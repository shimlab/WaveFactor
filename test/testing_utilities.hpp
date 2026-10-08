#ifndef TESTING_UTILITIES_INCLUDED
#define TESTING_UTILITIES_INCLUDED

#include "gtest/gtest.h"
#include "tensor.hpp"
#include "parameters.hpp"

// Macro to compare relative equality with rtol=1e-5 and underflow floor atol=1e-8
#define EXPECT_REL_EQ(val1, val2) \
    EXPECT_TRUE(testing_utilities::isCloseRel((val1), (val2), 1e-5, 1e-8)) \
        << "Expected relative equality: " #val1 " (" << (val1) << ") vs " #val2 " (" << (val2) << ")"

namespace testing_utilities
{
    inline bool isCloseRel(double a, double b, double rtol = 1e-5, double atol = 1e-8)
    {
        double diff = std::abs(a - b);
        double tol = atol + rtol * std::abs(b);
        return diff <= tol;
    }
}

// Functions to compare equivalency (up to relative tolerance and atol floor) of multidimensional tensors/vectors
void compareVectorDouble(const std::vector<double> &v1, const std::vector<double> &v2, double rtol = 1e-5, double atol = 1e-8);
void compareTensor1D(const Tensor1D &t1, const Tensor1D &t2, double rtol = 1e-5, double atol = 1e-8);
void compareTensor2D(const Tensor2D &t1, const Tensor2D &t2, double rtol = 1e-5, double atol = 1e-8);
void compareTensor3D(const Tensor3D &t1, const Tensor3D &t2, double rtol = 1e-5, double atol = 1e-8);
void compareTensor4D(const Tensor4D &t1, const Tensor4D &t2, double rtol = 1e-5, double atol = 1e-8);

// Class and function to throw an exception that prints out contents of a std::vector<double>
class VectorException : public std::runtime_error
{
public:
    // Constructor that takes a vector of doubles
    explicit VectorException(const std::vector<double> &vec);

    // Getter to access the stored vector
    const std::vector<double> &getVector() const;

private:
    // Member to hold the vector
    std::vector<double> vector_;

    // Helper function to format the error message
    static std::string formatMessage(const std::vector<double> &vec);
};

void throwVectorException(const std::vector<double> &vec);

#endif /*TESTING_UTILITIES_INCLUDED*/