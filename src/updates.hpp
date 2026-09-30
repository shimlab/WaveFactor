#ifndef UPDATES_HPP_INCLUDED
#define UPDATES_HPP_INCLUDED

#include <cmath>
#include "utilities.hpp"
#include "parameters.hpp"

// Global constants
extern const double RELATIVE_PMF_INCREMENT;
extern const double LOG_RELATIVE_PMF_INCREMENT;
extern const double EFFECTIVE_THRESHOLD;
extern const double EFFECTIVE_ZERO;
extern const double EFFECTIVE_ONE;
extern const double EFFECTIVE_LOG_ZERO;
extern const double EFFECTIVE_LOG_ONE;

// Parameter update grouping structs
struct UpdateLResult
{
    double sigma_squared_L;
    double mu_L;
    double log_r_L;
};

struct UpdateFResult
{
    double sigma_squared_F;
    double mu_F;
    double log_r_F;
};

struct UpdateTauResult
{
    double alpha_hat_tau;
    double beta_hat_tau;
};

struct UpdateTResult
{
    double alpha_hat_t;
    double beta_hat_t;
};

// For L_ijk_l, Z_L_ijk_m related updates
double compute_update_sigma_squared_L(int i, int j, int k, int l, const Parameters &parameters);
double compute_update_mu_L(int i, int j, int k, int l, double update_sigma_squared_L_ijk_l, const Parameters &parameters);
double compute_update_Z_L_log_relative_pmf(int i, int j, int k, int l, int z_L, double update_sigma_squared_L_ijk_l, double update_mu_L_ijk_l, const Parameters &parameters);
double compute_update_log_r_L(int i, int j, int k, int l, double update_sigma_squared_L_ijk_l, double update_mu_L_ijk_l, const Parameters &parameters);
UpdateLResult compute_update_L(int i, int j, int k, int l, const Parameters &parameters);

// For F_i_j, Z_F_i_j related updates
double compute_update_sigma_squared_F(int i, int j, const Parameters &parameters);
double compute_update_mu_F(int i, int j, double update_sigma_squared_F_i_j, const Parameters &parameters);
double compute_update_Z_F_log_relative_pmf(int i, int j, int z_F, double update_sigma_squared_F_i_j, double update_mu_F_i_j, const Parameters &parameters);
double compute_update_log_r_F(int i, int j, double update_sigma_squared_F_i_j, double update_mu_F_i_j, const Parameters &parameters);
UpdateFResult compute_update_F(int i, int j, const Parameters &parameters);

// For tau_i_l related updates
double compute_update_alpha_hat_tau(int i, int l, const Parameters &parameters);
double compute_update_beta_hat_tau(int i, int l, const Parameters &parameters);
UpdateTauResult compute_update_tau(int i, int l, const Parameters &parameters);
double compute_update_alpha_hat_t(int i, int l, const Parameters &parameters);
double compute_update_beta_hat_t(int i, int l, const Parameters &parameters);
UpdateTResult compute_update_t(int i, int l, const Parameters &parameters);

#endif /*UPDATES_HPP_INCLUDED*/