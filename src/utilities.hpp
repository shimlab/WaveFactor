#ifndef UTILITIES_HPP_INCLUDED
#define UTILITIES_HPP_INCLUDED

#include <cmath>
#include "parameters.hpp"
#include <unsupported/Eigen/SpecialFunctions>

struct LZLUpdateContext;
struct FZFTauUpdateContext;

inline double sum_log(double log_a, double log_b)
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

inline double gamma_t(int i, int l, const Parameters &parameters)
{
    double alpha_hat_t_i_l = parameters.alpha_hat_t[i][l];
    double beta_hat_t_i_l = parameters.beta_hat_t[i][l];
    return alpha_hat_t_i_l / beta_hat_t_i_l;
}

inline double gamma_tau(int i, int l, const Parameters &parameters)
{
    double alpha_hat_tau_i_l = parameters.alpha_hat_tau[i][l];
    double beta_hat_tau_i_l = parameters.beta_hat_tau[i][l];
    return alpha_hat_tau_i_l / beta_hat_tau_i_l;
}

inline double xi_L(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = parameters.r_L[l][i][j][k];
    double mu_L_ijk_l = parameters.mu_L[l][i][j][k];
    return r_L_ijk_l * mu_L_ijk_l;
}

inline double xi_F(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = parameters.r_F[i][j];
    double mu_F_i_j = parameters.mu_F[i][j];
    return r_F_i_j * mu_F_i_j;
}

inline double lambda_L(int i, int j, int k, int l, const Parameters &parameters)
{
    double r_L_ijk_l = parameters.r_L[l][i][j][k];
    double mu_L_ijk_l = parameters.mu_L[l][i][j][k];
    double sigma_squared_L_ijk_l = parameters.sigma_squared_L[l][i][j][k];
    return r_L_ijk_l * (sigma_squared_L_ijk_l + mu_L_ijk_l * mu_L_ijk_l);
}

inline double lambda_F(int i, int j, const Parameters &parameters)
{
    double r_F_i_j = parameters.r_F[i][j];
    double mu_F_i_j = parameters.mu_F[i][j];
    double sigma_squared_F_i_j = parameters.sigma_squared_F[i][j];
    return r_F_i_j * (sigma_squared_F_i_j + mu_F_i_j * mu_F_i_j);
}

inline double theta_t(int i, int l, const Parameters &parameters)
{
    double alpha_hat_t_i_l = parameters.alpha_hat_t[i][l];
    double beta_hat_t_i_l = parameters.beta_hat_t[i][l];
    return Eigen::numext::digamma(alpha_hat_t_i_l) - std::log(2 * M_PI * beta_hat_t_i_l);
}

inline double theta_tau(int i, int l, const Parameters &parameters)
{
    double alpha_hat_tau_i_l = parameters.alpha_hat_tau[i][l];
    double beta_hat_tau_i_l = parameters.beta_hat_tau[i][l];
    return Eigen::numext::digamma(alpha_hat_tau_i_l) - std::log(2 * M_PI * beta_hat_tau_i_l);
}

double u_F(int i, int l, int d, const Parameters &parameters);
double s_bar_F(int i, int j, int k, int l, const LZLUpdateContext &ctx_i_l, const Parameters &parameters);
double u_bar_F(int i, int l, const Parameters &parameters);
double u_L(int a, int b, int c, int i, int j, const Parameters &parameters);
double s_bar_L(int i, int j, const FZFTauUpdateContext &ctx, const Parameters &parameters);
double u_bar_L(int i, int j, const Parameters &parameters);

#endif /*UTILITIES_HPP_INCLUDED*/