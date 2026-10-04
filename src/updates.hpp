#ifndef UPDATES_HPP_INCLUDED
#define UPDATES_HPP_INCLUDED

#include <cmath>
#include <vector>
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

// Context holding resolution- and factor-level invariants for L and Z_L updates
struct LZLUpdateContext
{
    double update_sigma_squared_L;
    double log_scaling_factor;
    double log_bernoulli_true;
    double log_bernoulli_false;
    double theta_t_i_l;
    std::vector<double> nu_L;
    std::vector<double> phi_L;
    Tensor2D dot_Y;
};

// Context holding deduplicated spatial factor Gram matrix and data projections shared between F, Z_F and tau updates
struct FZFTauUpdateContext
{
    std::vector<std::vector<double>> lambda_bar_L;
    std::vector<std::vector<std::vector<double>>> phi_F;
    std::vector<std::vector<std::vector<double>>> nu_F;
};

FZFTauUpdateContext make_F_Z_F_tau_update_context(const Parameters &parameters);

// Factor-level context holding invariants for factor i across all features j
struct FZFUpdateContextForFactor
{
    std::vector<double> s_bar_L;
    std::vector<double> sigma_squared_F;
};

FZFUpdateContextForFactor make_F_Z_F_update_context_for_factor(int i, const FZFTauUpdateContext &ctx_F_Z_F_tau, const Parameters &parameters);

// Parameter update grouping structs
struct UpdateLZLResult
{
    double update_sigma_squared_L;
    double update_mu_L;
    double update_log_r_L;
};

struct UpdateFZFResult
{
    double update_sigma_squared_F;
    double update_mu_F;
    double update_log_r_F;
};

struct UpdateTauResult
{
    double update_alpha_hat_tau;
    double update_beta_hat_tau;
};

struct UpdateTResult
{
    double update_alpha_hat_t;
    double update_beta_hat_t;
};

// For L_ijk_l, Z_L_ijk_l related updates
LZLUpdateContext make_L_Z_L_update_context(int i, int l, const Parameters &parameters);
double compute_update_sigma_squared_L(int i, int j, int k, int l, const Parameters &parameters);
UpdateLZLResult compute_update_L_Z_L(int i, int j, int k, int l, const LZLUpdateContext &ctx_i_l, const Parameters &parameters);

// For F_i_j, Z_F_i_j related updates
UpdateFZFResult compute_update_F_Z_F(int i, int j, const FZFUpdateContextForFactor &ctx_factor_i, const Parameters &parameters);

// For tau_i_l related updates
double compute_update_alpha_hat_tau(int i, int l, const Parameters &parameters);
double compute_update_beta_hat_tau(int i, int l, const FZFTauUpdateContext &ctx, const Parameters &parameters);
UpdateTauResult compute_update_tau(int i, int l, const FZFTauUpdateContext &ctx, const Parameters &parameters);
double compute_update_alpha_hat_t(int i, int l, const Parameters &parameters);
double compute_update_beta_hat_t(int i, int l, const Parameters &parameters);
UpdateTResult compute_update_t(int i, int l, const Parameters &parameters);

#endif /*UPDATES_HPP_INCLUDED*/