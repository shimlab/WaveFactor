#include "parameters.hpp"
#include <cmath>
#include <limits>

// Constructor to initialize all fields based on provided values
Parameters::Parameters(int n_resolutions_init, int n_factors_init, int n_features_init,
                       const Tensor4D &Y_init,
                       const Tensor1D &log_pi_L_init,
                       const Tensor2D &log_pi_F_init,
                       const Tensor2D &alpha_t_init,
                       const Tensor2D &beta_t_init,
                       const Tensor2D &alpha_tau_init,
                       const Tensor2D &beta_tau_init,
                       const Tensor4D &mu_L_init,
                       const Tensor4D &sigma_squared_L_init,
                       const Tensor4D &log_r_L_init,
                       const Tensor2D &mu_F_init,
                       const Tensor2D &sigma_squared_F_init,
                       const Tensor2D &log_r_F_init,
                       const Tensor2D &alpha_hat_t_init,
                       const Tensor2D &beta_hat_t_init,
                       const Tensor2D &alpha_hat_tau_init,
                       const Tensor2D &beta_hat_tau_init)
    : n_resolutions(n_resolutions_init), n_factors(n_factors_init), n_features(n_features_init),
      Y(Y_init), log_pi_L(log_pi_L_init), log_pi_F(log_pi_F_init),
      alpha_t(alpha_t_init), beta_t(beta_t_init),
      alpha_tau(alpha_tau_init), beta_tau(beta_tau_init),
      mu_L(mu_L_init), sigma_squared_L(sigma_squared_L_init), log_r_L(log_r_L_init),
      mu_F(mu_F_init), sigma_squared_F(sigma_squared_F_init), log_r_F(log_r_F_init),
      alpha_hat_t(alpha_hat_t_init), beta_hat_t(beta_hat_t_init),
       alpha_hat_tau(alpha_hat_tau_init), beta_hat_tau(beta_hat_tau_init)
{
    // Initialize cached r_L = exp(log_r_L)
    r_L = log_r_L;
    for (size_t l = 0; l < r_L.size(); ++l)
    {
        for (size_t i = 0; i < r_L[l].size(); ++i)
        {
            for (size_t j = 0; j < r_L[l][i].size(); ++j)
            {
                for (size_t k = 0; k < r_L[l][i][j].size(); ++k)
                {
                    r_L[l][i][j][k] = std::exp(log_r_L[l][i][j][k]);
                }
            }
        }
    }

    // Initialize cached r_F = exp(log_r_F)
    r_F = log_r_F;
    for (size_t i = 0; i < r_F.size(); ++i)
    {
        for (size_t j = 0; j < r_F[i].size(); ++j)
        {
            r_F[i][j] = std::exp(log_r_F[i][j]);
        }
    }

    // Initialize cached N_coefs_per_res per resolution level i with signaling NaN for defensive poisoning
    N_coefs_per_res.assign(n_resolutions, std::numeric_limits<double>::signaling_NaN());
    for (size_t i = 0; i < N_coefs_per_res.size(); ++i)
    {
        double sum_n = 0.0;
        for (size_t j = 0; j < Y[0][i].size(); ++j)
        {
            sum_n += Y[0][i][j].size();
        }
        N_coefs_per_res[i] = sum_n;
    }
}

// Copy constructor for deep copying
Parameters::Parameters(const Parameters &other)
    : n_resolutions(other.n_resolutions),
      n_factors(other.n_factors),
      n_features(other.n_features),
      Y(other.Y),
      log_pi_L(other.log_pi_L),
      log_pi_F(other.log_pi_F),
      alpha_t(other.alpha_t),
      beta_t(other.beta_t),
      alpha_tau(other.alpha_tau),
      beta_tau(other.beta_tau),
      mu_L(other.mu_L),
      sigma_squared_L(other.sigma_squared_L),
      log_r_L(other.log_r_L),
      r_L(other.r_L),
      mu_F(other.mu_F),
      sigma_squared_F(other.sigma_squared_F),
      log_r_F(other.log_r_F),
      r_F(other.r_F),
      alpha_hat_t(other.alpha_hat_t),
      beta_hat_t(other.beta_hat_t),
      alpha_hat_tau(other.alpha_hat_tau),
      beta_hat_tau(other.beta_hat_tau),
      N_coefs_per_res(other.N_coefs_per_res) {}

// Copy assignment operator for deep copying
Parameters &Parameters::operator=(const Parameters &other)
{
    if (this == &other)
        return *this; // Handle self-assignment

    n_resolutions = other.n_resolutions;
    n_factors = other.n_factors;
    n_features = other.n_features;
    Y = other.Y;
    log_pi_L = other.log_pi_L;
    log_pi_F = other.log_pi_F;
    alpha_t = other.alpha_t;
    beta_t = other.beta_t;
    alpha_tau = other.alpha_tau;
    beta_tau = other.beta_tau;
    mu_L = other.mu_L;
    sigma_squared_L = other.sigma_squared_L;
    log_r_L = other.log_r_L;
    r_L = other.r_L;
    mu_F = other.mu_F;
    sigma_squared_F = other.sigma_squared_F;
    log_r_F = other.log_r_F;
    r_F = other.r_F;
    alpha_hat_t = other.alpha_hat_t;
    beta_hat_t = other.beta_hat_t;
    alpha_hat_tau = other.alpha_hat_tau;
    beta_hat_tau = other.beta_hat_tau;
    N_coefs_per_res = other.N_coefs_per_res;

    return *this;
}