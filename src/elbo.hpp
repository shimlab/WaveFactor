#ifndef ELBO_HPP_INCLUDED
#define ELBO_HPP_INCLUDED

#include "parameters.hpp"
#include <cmath>
#include "utilities.hpp"
#include <unsupported/Eigen/SpecialFunctions>

inline double compute_E_log_likelihood_Y_ijk_l_given_pi_L_F_tau(int i, int j, int k, int l, const Parameters &parameters)
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

inline double compute_E_log_likelihood_L_ijk_l_given_Z_L_t(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = parameters.r_L[l][i][j][k];
    double theta_t_i_l = theta_t(i, l, parameters);
    double gamma_t_i_l = gamma_t(i, l, parameters);
    double lambda_L_ijk_l = lambda_L(i, j, k, l, parameters);
    return 0.5 * (r_L_ijk_l * theta_t_i_l - gamma_t_i_l * lambda_L_ijk_l);
}

inline double compute_E_log_likelihood_F_i_j_given_Z_F(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = parameters.r_F[i][j];
    double lambda_F_i_j = lambda_F(i, j, parameters);
    return -0.5 * (r_F_i_j * std::log(2 * M_PI) + lambda_F_i_j);
}

inline double compute_E_log_likelihood_Z_L_ijk_l(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = parameters.r_L[l][i][j][k];
    double log_pi_L_i = parameters.log_pi_L[i];
    double pi_L_i = std::exp(log_pi_L_i);
    return r_L_ijk_l * log_pi_L_i + (1 - r_L_ijk_l) * std::log(1 - pi_L_i);
}

inline double compute_E_log_likelihood_Z_F_i_j(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = parameters.r_F[i][j];
    double log_pi_F_i_j = parameters.log_pi_F[i][j];
    double pi_F_i_j = std::exp(log_pi_F_i_j);
    return r_F_i_j * log_pi_F_i_j + (1 - r_F_i_j) * std::log(1 - pi_F_i_j);
}

inline double compute_E_log_likelihood_t_i_l(int i, int l, const Parameters &parameters)
{
    double alpha_t_i_l = parameters.alpha_t[i][l];
    double beta_t_i_l = parameters.beta_t[i][l];
    double alpha_hat_t_i_l = parameters.alpha_hat_t[i][l];
    double beta_hat_t_i_l = parameters.beta_hat_t[i][l];
    double gamma_t_i_l = gamma_t(i, l, parameters);
    return (alpha_t_i_l - 1) * (Eigen::numext::digamma(alpha_hat_t_i_l) - std::log(beta_hat_t_i_l)) - gamma_t_i_l * beta_t_i_l +
           alpha_t_i_l * std::log(beta_t_i_l) - std::lgamma(alpha_t_i_l);
}

inline double compute_E_log_likelihood_tau_i_l(int i, int l, const Parameters &parameters)
{
    double alpha_tau_i_l = parameters.alpha_tau[i][l];
    double beta_tau_i_l = parameters.beta_tau[i][l];
    double alpha_hat_tau_i_l = parameters.alpha_hat_tau[i][l];
    double beta_hat_tau_i_l = parameters.beta_hat_tau[i][l];
    double gamma_tau_i_l = gamma_tau(i, l, parameters);
    return (alpha_tau_i_l - 1) * (Eigen::numext::digamma(alpha_hat_tau_i_l) - std::log(beta_hat_tau_i_l)) - gamma_tau_i_l * beta_tau_i_l +
           alpha_tau_i_l * std::log(beta_tau_i_l) - std::lgamma(alpha_tau_i_l);
}

inline double compute_E_negative_variational_log_likelihood_L_ijk_l_Z_L_ijk_l(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = parameters.r_L[l][i][j][k];
    double sigma_squared_L_ijk_l = parameters.sigma_squared_L[l][i][j][k];
    return (r_L_ijk_l / 2) * (std::log(2 * M_PI * sigma_squared_L_ijk_l) + 1) - r_L_ijk_l * std::log(r_L_ijk_l) -
           (1 - r_L_ijk_l) * std::log(1 - r_L_ijk_l);
}

inline double compute_E_negative_variational_log_likelihood_F_i_j_Z_F_i_j(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = parameters.r_F[i][j];
    double sigma_squared_F_i_j = parameters.sigma_squared_F[i][j];
    return (r_F_i_j / 2) * (std::log(2 * M_PI * sigma_squared_F_i_j) + 1) - r_F_i_j * std::log(r_F_i_j) -
           (1 - r_F_i_j) * std::log(1 - r_F_i_j);
}

inline double compute_E_negative_variational_log_likelihood_t_i_l(int i, int l, const Parameters &parameters)
{
    double alpha_hat_t_i_l = parameters.alpha_hat_t[i][l];
    double beta_hat_t_i_l = parameters.beta_hat_t[i][l];
    return alpha_hat_t_i_l - std::log(beta_hat_t_i_l) + std::lgamma(alpha_hat_t_i_l) +
           (1 - alpha_hat_t_i_l) * Eigen::numext::digamma(alpha_hat_t_i_l);
}

inline double compute_E_negative_variational_log_likelihood_tau_i_l(int i, int l, const Parameters &parameters)
{
    double alpha_hat_tau_i_l = parameters.alpha_hat_tau[i][l];
    double beta_hat_tau_i_l = parameters.beta_hat_tau[i][l];
    return alpha_hat_tau_i_l - std::log(beta_hat_tau_i_l) + std::lgamma(alpha_hat_tau_i_l) +
           (1 - alpha_hat_tau_i_l) * Eigen::numext::digamma(alpha_hat_tau_i_l);
}

inline double compute_elbo(bool assume_tau_updated, const Parameters &parameters)
{
    double elbo = 0;

    int n_factors = parameters.n_factors;
    int n_features = parameters.n_features;

    // Loop 1: F and Z_F
    for (int i = 0; i < n_factors; ++i)
    {
        for (int j = 0; j < n_features; ++j)
        {
            elbo += compute_E_log_likelihood_F_i_j_given_Z_F(i, j, parameters);
            elbo += compute_E_log_likelihood_Z_F_i_j(i, j, parameters);
            elbo += compute_E_negative_variational_log_likelihood_F_i_j_Z_F_i_j(i, j, parameters);
        }
    }

    // Loop 2: t
    for (int l = 0; l < n_factors; ++l)
    {
        for (int i = 0; i < parameters.mu_L[l].size(); ++i)
        {
            elbo += compute_E_log_likelihood_t_i_l(i, l, parameters);
            elbo += compute_E_negative_variational_log_likelihood_t_i_l(i, l, parameters);
        }
    }

    // Loop 3: tau
    for (int l = 0; l < n_features; ++l)
    {
        for (int i = 0; i < parameters.Y[l].size(); ++i)
        {
            elbo += compute_E_log_likelihood_tau_i_l(i, l, parameters);
            elbo += compute_E_negative_variational_log_likelihood_tau_i_l(i, l, parameters);
        }
    }

    // Loop 4: L and Z_L
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

    // Loop 5: Expected Log-Likelihood of Y
    if (assume_tau_updated)
    {
        for (int l = 0; l < n_features; ++l)
        {
            for (int i = 0; i < parameters.Y[l].size(); ++i)
            {
                double N_i = static_cast<double>(parameters.wavelet_indices[i].n_coefficients);
                double theta_tau_i_l = theta_tau(i, l, parameters);
                double gamma_tau_i_l = gamma_tau(i, l, parameters);
                double delta_beta = parameters.beta_hat_tau[i][l] - parameters.beta_tau[i][l];

                elbo += 0.5 * N_i * theta_tau_i_l - gamma_tau_i_l * delta_beta;
            }
        }
    }
    else
    {
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
    }

    return elbo;
}

#endif /*ELBO_HPP_INCLUDED*/