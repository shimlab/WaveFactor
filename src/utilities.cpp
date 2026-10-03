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

double u_L(int a, int b, int c, int i, int j, const Parameters &parameters)
{
    double lambda_L_abc_i = lambda_L(a, b, c, i, parameters);
    double gamma_tau_a_j = gamma_tau(a, j, parameters);
    return lambda_L_abc_i * gamma_tau_a_j;
}

double s_bar_L(int i, int j, const FZFUpdateContext &ctx_i, const Parameters &parameters)
{
    double s_sum = 0.0;
    for (int a = 0; a < parameters.n_resolutions; ++a)
    {
        double dot_phi = 0.0;
        for (int m = 0; m < parameters.n_factors; ++m)
        {
            dot_phi += ctx_i.phi_F[a][m] * xi_F(m, j, parameters);
        }
        s_sum += gamma_tau(a, j, parameters) * (ctx_i.nu_F[a][j] - dot_phi);
    }
    return s_sum;
}

double u_bar_L(int i, int j, const Parameters &parameters)
{
    double u_sum = 0.0;
    for (int a = 0; a < parameters.n_resolutions; ++a)
    {
        int max_b = parameters.mu_L[0][a].size();
        for (int b = 0; b < max_b; ++b)
        {
            int max_c = parameters.mu_L[0][a][b].size();
            for (int c = 0; c < max_c; ++c)
            {
                u_sum += u_L(a, b, c, i, j, parameters);
            }
        }
    }
    return u_sum;
}