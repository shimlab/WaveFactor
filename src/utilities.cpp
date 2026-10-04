#include "utilities.hpp"
#include "updates.hpp"

// Utility functions
double u_F(int i, int l, int d, const Parameters &parameters)
{
    double lambda_F_l_d = lambda_F(l, d, parameters);
    double gamma_tau_i_d = gamma_tau(i, d, parameters);
    return lambda_F_l_d * gamma_tau_i_d;
}

double s_bar_F(int i, int j, int k, int l, const LZLUpdateContext &ctx_i_l, const Parameters &parameters)
{
    double dot_Y = ctx_i_l.dot_Y[j][k];

    double dot_phi = 0.0;
    for (int m = 0; m < parameters.n_factors; ++m)
    {
        if (m == l) continue;
        dot_phi += ctx_i_l.phi_L[m] * xi_L(i, j, k, m, parameters);
    }

    return dot_Y - dot_phi;
}

double u_bar_F(int i, int l, const Parameters &parameters)
{
    double u_sum = 0.0;
    for (int d = 0; d < parameters.n_features; ++d)
    {
        u_sum += u_F(i, l, d, parameters);
    }
    return u_sum;
}