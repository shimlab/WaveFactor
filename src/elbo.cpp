#include "elbo.hpp"

double compute_E_log_likelihood_Y_ijk_l_given_pi_L_F_tau(int i, int j, int k, int l, const Parameters &parameters)
{
    double theta_tau_i_l = theta_tau(i, l, parameters);
    double gamma_tau_i_l = gamma_tau(i, l, parameters);
    double Y_ijk_l = parameters.Y[l][i][j][k];

    int max_m = parameters.n_factors; // i.e. number of factors

    double xi_product_sum = 0.0;
    double xi_product_sq_sum = 0.0;
    double lambda_product_sum = 0.0;
    for (int m = 0; m < max_m; ++m)
    {
        double x_m = xi_L(i, j, k, m, parameters) * xi_F(m, l, parameters);
        xi_product_sum += x_m;
        xi_product_sq_sum += x_m * x_m;
        lambda_product_sum += lambda_L(i, j, k, m, parameters) * lambda_F(m, l, parameters);
    }

    double xi_quad_product_sum = xi_product_sum * xi_product_sum - xi_product_sq_sum;

    return 0.5 * (theta_tau_i_l - gamma_tau_i_l * (Y_ijk_l * Y_ijk_l - 2 * Y_ijk_l * xi_product_sum +
                                                   xi_quad_product_sum + lambda_product_sum));
}

double compute_elbo(const Parameters &parameters)
{
    double elbo = 0;

    // Summing terms across factors and features
    int n_factors = parameters.n_factors;
    int n_features = parameters.n_features;

    // Summing terms across factors and features
    for (int i = 0; i < n_factors; ++i)
    {
        for (int j = 0; j < n_features; ++j)
        {
            elbo += compute_E_log_likelihood_F_i_j_given_Z_F(i, j, parameters);
            elbo += compute_E_log_likelihood_Z_F_i_j(i, j, parameters);
            elbo += compute_E_negative_variational_log_likelihood_F_i_j_Z_F_i_j(i, j, parameters);
        }
    }

    // Summing terms across wavelet resolution and factors
    for (int l = 0; l < n_factors; ++l)
    {
        for (int i = 0; i < parameters.mu_L[l].size(); ++i)
        {
            elbo += compute_E_log_likelihood_t_i_l(i, l, parameters);
            elbo += compute_E_negative_variational_log_likelihood_t_i_l(i, l, parameters);
        }
    }

    // Summing terms across wavelet resolution and features
    for (int l = 0; l < n_features; ++l)
    {
        for (int i = 0; i < parameters.Y[l].size(); ++i)
        {
            elbo += compute_E_log_likelihood_tau_i_l(i, l, parameters);
            elbo += compute_E_negative_variational_log_likelihood_tau_i_l(i, l, parameters);
        }
    }

    // Summing terms across wavelet coefficients and factors
    for (int l = 0; l < n_factors; ++l)
    {
        for (int i = 0; i < parameters.mu_L[l].size(); ++i)
        {
            for (int j = 0; j < parameters.mu_L[l][i].size(); ++j)
            {
                for (int k = 0; k < parameters.mu_L[l][i][j].size(); ++k)
                {
                    elbo += compute_E_log_likelihood_L_ijk_l_given_Z_L_t(i, j, k, l, parameters);
                    elbo += compute_E_log_likelihood_Z_L_ijk_l(i, j, k, l, parameters);
                    elbo += compute_E_negative_variational_log_likelihood_L_ijk_l_Z_L_ijk_l(i, j, k, l, parameters);
                }
            }
        }
    }

    // Summing terms across wavelet coefficients and features
    for (int l = 0; l < n_features; ++l)
    {
        for (int i = 0; i < parameters.Y[l].size(); ++i)
        {
            for (int j = 0; j < parameters.Y[l][i].size(); ++j)
            {
                for (int k = 0; k < parameters.Y[l][i][j].size(); ++k)
                {
                    elbo += compute_E_log_likelihood_Y_ijk_l_given_pi_L_F_tau(i, j, k, l, parameters);
                }
            }
        }
    }

    return elbo;
}
