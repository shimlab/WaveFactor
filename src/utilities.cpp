#include "utilities.hpp"
#include "updates.hpp"

// Helper functions
double sum_log(double log_a, double log_b)
{
    // Computes log(a+b) given log(a) and log(b)
    if (log_a > log_b)
    {
        return log_a + std::log(1 + std::exp(log_b - log_a));
    }
    else
    {
        return log_b + std::log(std::exp(log_a - log_b) + 1);
    }
}

// Utility functions
double gamma_t(int i, int l, const Parameters &parameters)
{
    double alpha_hat_t_i_l = parameters.alpha_hat_t[i][l];
    double beta_hat_t_i_l = parameters.beta_hat_t[i][l];
    return alpha_hat_t_i_l / beta_hat_t_i_l;
}

double gamma_tau(int i, int l, const Parameters &parameters)
{
    double alpha_hat_tau_i_l = parameters.alpha_hat_tau[i][l];
    double beta_hat_tau_i_l = parameters.beta_hat_tau[i][l];
    return alpha_hat_tau_i_l / beta_hat_tau_i_l;
}

double xi_L(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = parameters.r_L[l][i][j][k];
    double mu_L_ijk_l = parameters.mu_L[l][i][j][k];
    return r_L_ijk_l * mu_L_ijk_l;
}

double xi_F(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = parameters.r_F[i][j];
    double mu_F_i_j = parameters.mu_F[i][j];
    return r_F_i_j * mu_F_i_j;
}

double lambda_L(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = parameters.r_L[l][i][j][k];
    double mu_L_ijk_l = parameters.mu_L[l][i][j][k];
    double sigma_squared_L_ijk_l = parameters.sigma_squared_L[l][i][j][k];
    return r_L_ijk_l * (sigma_squared_L_ijk_l + mu_L_ijk_l * mu_L_ijk_l);
}

double lambda_F(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = parameters.r_F[i][j];
    double mu_F_i_j = parameters.mu_F[i][j];
    double sigma_squared_F_i_j = parameters.sigma_squared_F[i][j];
    return r_F_i_j * (sigma_squared_F_i_j + mu_F_i_j * mu_F_i_j);
}

double theta_t(int i, int l, const Parameters &parameters)
{
    double alpha_hat_t_i_l = parameters.alpha_hat_t[i][l];
    double beta_hat_t_i_l = parameters.beta_hat_t[i][l];
    return Eigen::numext::digamma(alpha_hat_t_i_l) - std::log(2 * M_PI * beta_hat_t_i_l);
}

double theta_tau(int i, int l, const Parameters &parameters)
{
    double alpha_hat_tau_i_l = parameters.alpha_hat_tau[i][l];
    double beta_hat_tau_i_l = parameters.beta_hat_tau[i][l];
    return Eigen::numext::digamma(alpha_hat_tau_i_l) - std::log(2 * M_PI * beta_hat_tau_i_l);
}

double u_F(int i, int l, int d, const Parameters &parameters)
{
    double lambda_F_l_d = lambda_F(l, d, parameters);
    double gamma_tau_i_d = gamma_tau(i, d, parameters);
    return lambda_F_l_d * gamma_tau_i_d;
}



double s_bar_F(int i, int j, int k, int l, const LZLUpdateContext &ctx_i_l, const Parameters &parameters)
{
    double dot_Y = 0.0;
    for (int d = 0; d < parameters.n_features; ++d)
    {
        dot_Y += ctx_i_l.nu_L[d] * parameters.Y[d][i][j][k];
    }

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