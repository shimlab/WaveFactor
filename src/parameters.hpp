#ifndef PARAMETERS_HPP_INCLUDED
#define PARAMETERS_HPP_INCLUDED

#include "tensor.hpp"

class Parameters
{
public:
    int n_resolutions;
    int n_factors;
    int n_features;
    Tensor4D Y;
    Tensor1D log_pi_L;
    Tensor2D log_pi_F;
    Tensor2D alpha_t;
    Tensor2D beta_t;
    Tensor2D alpha_tau;
    Tensor2D beta_tau;
    Tensor4D mu_L;
    Tensor4D sigma_squared_L;
    Tensor4D log_r_L;
    Tensor4D r_L;
    Tensor2D mu_F;
    Tensor2D sigma_squared_F;
    Tensor2D log_r_F;
    Tensor2D r_F;
    Tensor2D alpha_hat_t;
    Tensor2D beta_hat_t;
    Tensor2D alpha_hat_tau;
    Tensor2D beta_hat_tau;
    Tensor1D N_coefs_per_res;

    // Constructor to initialize all fields based on provided values
    Parameters(int n_resolutions_init, int n_factors_init, int n_features_init,
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
               const Tensor2D &beta_hat_tau_init);

    // Copy constructor for deep copying
    Parameters(const Parameters &other);

    // Copy assignment operator for deep copying
    Parameters &operator=(const Parameters &other);
};

#endif /*PARAMETERS_HPP_INCLUDED*/