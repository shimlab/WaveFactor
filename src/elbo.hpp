#ifndef ELBO_HPP_INCLUDED
#define ELBO_HPP_INCLUDED

#include "parameters.hpp"
#include <cmath>
#include "utilities.hpp"
#include <unsupported/Eigen/SpecialFunctions>

double compute_E_log_likelihood_Y_ijk_l_given_pi_L_F_tau(int i, int j, int k, int l, const Parameters &parameters);
inline double compute_E_log_likelihood_L_ijk_l_given_Z_L_t(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = std::exp(parameters.log_r_L[l][i][j][k]);
    double theta_t_i_l = theta_t(i, l, parameters);
    double gamma_t_i_l = gamma_t(i, l, parameters);
    double lambda_L_ijk_l = lambda_L(i, j, k, l, parameters);
    return 0.5 * (r_L_ijk_l * theta_t_i_l - gamma_t_i_l * lambda_L_ijk_l);
}

inline double compute_E_log_likelihood_F_i_j_given_Z_F(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = std::exp(parameters.log_r_F[i][j]);
    double lambda_F_i_j = lambda_F(i, j, parameters);
    return -0.5 * (r_F_i_j * std::log(2 * M_PI) + lambda_F_i_j);
}

inline double compute_E_log_likelihood_Z_L_ijk_l(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = std::exp(parameters.log_r_L[l][i][j][k]);
    double log_pi_L_i = parameters.log_pi_L[i];
    double pi_L_i = std::exp(log_pi_L_i);
    return r_L_ijk_l * log_pi_L_i + (1 - r_L_ijk_l) * std::log(1 - pi_L_i);
}

inline double compute_E_log_likelihood_Z_F_i_j(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = std::exp(parameters.log_r_F[i][j]);
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

inline double compute_E_negative_variational_log_likelihood_L_ijk_l(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = std::exp(parameters.log_r_L[l][i][j][k]);
    double sigma_squared_L_ijk_l = parameters.sigma_squared_L[l][i][j][k];
    return (r_L_ijk_l / 2) * (std::log(2 * M_PI * sigma_squared_L_ijk_l) + 1) - r_L_ijk_l * std::log(r_L_ijk_l) -
           (1 - r_L_ijk_l) * std::log(1 - r_L_ijk_l);
}

inline double compute_E_negative_variational_log_likelihood_F_i_j(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = std::exp(parameters.log_r_F[i][j]);
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
double compute_elbo(const Parameters &parameters);

#endif /*ELBO_HPP_INCLUDED*/