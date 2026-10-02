#include "updates.hpp"
#include <limits>

// Global constants
const double RELATIVE_PMF_INCREMENT = 1e-10;
const double LOG_RELATIVE_PMF_INCREMENT = std::log(RELATIVE_PMF_INCREMENT);
const double EFFECTIVE_THRESHOLD = 1e-8;
const double EFFECTIVE_ZERO = EFFECTIVE_THRESHOLD;
const double EFFECTIVE_ONE = 1.0 - EFFECTIVE_THRESHOLD;
const double EFFECTIVE_LOG_ZERO = std::log(EFFECTIVE_ZERO);
const double EFFECTIVE_LOG_ONE = std::log(EFFECTIVE_ONE);

// For L_ijk_l, Z_L_ijk_l related updates
double compute_update_sigma_squared_L(int i, int j, int k, int l, const Parameters &parameters)
{
    double gamma_t_i_l = gamma_t(i, l, parameters);
    double u_bar_F_i_l = u_bar_F(i, l, parameters);
    return 1.0 / (gamma_t_i_l + u_bar_F_i_l);
}

double compute_update_mu_L(int i, int j, int k, int l, double update_sigma_squared_L_ijk_l, const Parameters &parameters)
{
    double s_bar_F_ijk_l = s_bar_F(i, j, k, l, parameters);
    return s_bar_F_ijk_l * update_sigma_squared_L_ijk_l;
}

double compute_update_Z_L_log_relative_pmf(int i, int j, int k, int l, int z_L, double update_sigma_squared_L_ijk_l, double update_mu_L_ijk_l, const Parameters &parameters)
{
    double log_pi_L_i = parameters.log_pi_L[i];
    double pi_L_i = exp(log_pi_L_i);

    double log_scaling_factor = 0;
    double log_exp_factor = 0;
    double log_bernoulli_factor = (z_L == 1) ? log_pi_L_i : log(1 - pi_L_i);
    if (z_L == 1)
    {
        double theta_t_i_l = theta_t(i, l, parameters);
        log_scaling_factor = 0.5 * (log(2) + log(M_PI) + log(update_sigma_squared_L_ijk_l));
        log_exp_factor = 0.5 * (theta_t_i_l + update_mu_L_ijk_l * update_mu_L_ijk_l / update_sigma_squared_L_ijk_l);
    }

    return log_scaling_factor + log_exp_factor + log_bernoulli_factor;
}

double compute_update_log_r_L(int i, int j, int k, int l, double update_sigma_squared_L_ijk_l, double update_mu_L_ijk_l, const Parameters &parameters)
{
    double relative_true_log_prob = compute_update_Z_L_log_relative_pmf(i, j, k, l, 1, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters);
    double relative_false_log_prob = compute_update_Z_L_log_relative_pmf(i, j, k, l, 0, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters);

    double true_log_prob = relative_true_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);
    double false_log_prob = relative_false_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);

    double incremented_true_log_prob = sum_log(true_log_prob, LOG_RELATIVE_PMF_INCREMENT);
    double incremented_false_log_prob = sum_log(false_log_prob, LOG_RELATIVE_PMF_INCREMENT);

    return incremented_true_log_prob - sum_log(incremented_true_log_prob, incremented_false_log_prob);
}

LZLUpdateContext make_L_Z_L_update_context(int i, int l, const Parameters &parameters)
{
    LZLUpdateContext ctx_i_l;
    ctx_i_l.update_sigma_squared_L = compute_update_sigma_squared_L(i, 0, 0, l, parameters);
    ctx_i_l.log_scaling_factor = 0.5 * (std::log(2.0) + std::log(M_PI) + std::log(ctx_i_l.update_sigma_squared_L));
    ctx_i_l.log_bernoulli_true = parameters.log_pi_L[i];
    ctx_i_l.log_bernoulli_false = std::log(1.0 - std::exp(parameters.log_pi_L[i]));
    ctx_i_l.theta_t_i_l = theta_t(i, l, parameters);

    ctx_i_l.nu_L.assign(parameters.n_features, std::numeric_limits<double>::signaling_NaN());
    for (int d = 0; d < parameters.n_features; ++d)
    {
        ctx_i_l.nu_L[d] = gamma_tau(i, d, parameters) * xi_F(l, d, parameters);
    }

    ctx_i_l.phi_L.assign(parameters.n_factors, std::numeric_limits<double>::signaling_NaN());
    for (int m = 0; m < parameters.n_factors; ++m)
    {
        if (m == l)
        {
            ctx_i_l.phi_L[m] = 0.0;
            continue;
        }
        double sum_m = 0.0;
        for (int d = 0; d < parameters.n_features; ++d)
        {
            sum_m += ctx_i_l.nu_L[d] * xi_F(m, d, parameters);
        }
        ctx_i_l.phi_L[m] = sum_m;
    }

    return ctx_i_l;
}

UpdateLZLResult compute_update_L_Z_L(int i, int j, int k, int l, const LZLUpdateContext &ctx_i_l, const Parameters &parameters)
{
    double s_bar_F_ijk_l = s_bar_F(i, j, k, l, ctx_i_l, parameters);
    double update_mu_L_ijk_l = s_bar_F_ijk_l * ctx_i_l.update_sigma_squared_L;

    double relative_true_log_prob = ctx_i_l.log_bernoulli_true + ctx_i_l.log_scaling_factor +
        0.5 * (ctx_i_l.theta_t_i_l + (update_mu_L_ijk_l * update_mu_L_ijk_l) / ctx_i_l.update_sigma_squared_L);
    double relative_false_log_prob = ctx_i_l.log_bernoulli_false;

    double true_log_prob = relative_true_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);
    double false_log_prob = relative_false_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);

    double incremented_true_log_prob = sum_log(true_log_prob, LOG_RELATIVE_PMF_INCREMENT);
    double incremented_false_log_prob = sum_log(false_log_prob, LOG_RELATIVE_PMF_INCREMENT);

    double update_log_r_L_ijk_l = incremented_true_log_prob - sum_log(incremented_true_log_prob, incremented_false_log_prob);

    return {
        ctx_i_l.update_sigma_squared_L,
        update_mu_L_ijk_l,
        update_log_r_L_ijk_l};
}

UpdateLZLResult compute_update_L_Z_L(int i, int j, int k, int l, const Parameters &parameters)
{
    double update_sigma_squared_L_ijk_l = compute_update_sigma_squared_L(i, j, k, l, parameters);
    double update_mu_L_ijk_l = compute_update_mu_L(i, j, k, l, update_sigma_squared_L_ijk_l, parameters);
    double update_log_r_L_ijk_l = compute_update_log_r_L(i, j, k, l, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters);

    return {
        update_sigma_squared_L_ijk_l,
        update_mu_L_ijk_l,
        update_log_r_L_ijk_l};
}

// For F_i_j, Z_F_i_j related updates
double compute_update_sigma_squared_F(int i, int j, const Parameters &parameters)
{
    double u_bar_L_i_j = u_bar_L(i, j, parameters);
    return 1.0 / (1 + u_bar_L_i_j);
}

double compute_update_mu_F(int i, int j, double update_sigma_squared_F_i_j, const Parameters &parameters)
{
    double s_bar_L_i_j = s_bar_L(i, j, parameters);
    return s_bar_L_i_j * update_sigma_squared_F_i_j;
}

double compute_update_Z_F_log_relative_pmf(int i, int j, int z_F, double update_sigma_squared_F_i_j, double update_mu_F_i_j, const Parameters &parameters)
{
    double log_pi_F_i_j = parameters.log_pi_F[i][j];
    double pi_F_i_j = exp(log_pi_F_i_j);

    double log_scaling_factor = 0;
    double log_exp_factor = 0;
    double log_bernoulli_factor = (z_F == 1) ? log_pi_F_i_j : log(1 - pi_F_i_j);
    if (z_F == 1)
    {
        log_scaling_factor = 0.5 * (log(2) + log(M_PI) + log(update_sigma_squared_F_i_j));
        log_exp_factor = 0.5 * (-log(2 * M_PI) + update_mu_F_i_j * update_mu_F_i_j / update_sigma_squared_F_i_j);
    }

    return log_scaling_factor + log_exp_factor + log_bernoulli_factor;
}

double compute_update_log_r_F(int i, int j, double update_sigma_squared_F_i_j, double update_mu_F_i_j, const Parameters &parameters)
{
    double relative_true_log_prob = compute_update_Z_F_log_relative_pmf(i, j, 1, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters);
    double relative_false_log_prob = compute_update_Z_F_log_relative_pmf(i, j, 0, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters);

    double true_log_prob = relative_true_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);
    double false_log_prob = relative_false_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);

    double incremented_true_log_prob = sum_log(true_log_prob, LOG_RELATIVE_PMF_INCREMENT);
    double incremented_false_log_prob = sum_log(false_log_prob, LOG_RELATIVE_PMF_INCREMENT);

    return incremented_true_log_prob - sum_log(incremented_true_log_prob, incremented_false_log_prob);
}

UpdateFZFResult compute_update_F_Z_F(int i, int j, const Parameters &parameters)
{
    double update_sigma_squared_F_i_j = compute_update_sigma_squared_F(i, j, parameters);
    double update_mu_F_i_j = compute_update_mu_F(i, j, update_sigma_squared_F_i_j, parameters);
    double log_pi_F_i_j = parameters.log_pi_F[i][j];
    double update_log_r_F_i_j;
    if (log_pi_F_i_j < EFFECTIVE_LOG_ZERO) {
      update_log_r_F_i_j = EFFECTIVE_LOG_ZERO;
    } else if (log_pi_F_i_j > EFFECTIVE_LOG_ONE) {
      update_log_r_F_i_j = EFFECTIVE_LOG_ONE;
    } else {
      update_log_r_F_i_j = compute_update_log_r_F(i, j, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters);
    }

    return {
        update_sigma_squared_F_i_j,
        update_mu_F_i_j,
        update_log_r_F_i_j};
}

// For tau_i_l related updates
double compute_update_alpha_hat_tau(int i, int l, const Parameters &parameters)
{
    double alpha_tau_i_l = parameters.alpha_tau[i][l];
    double N_i = 0.0;
    for (size_t j = 0; j < parameters.Y[l][i].size(); ++j)
    {
        N_i += parameters.Y[l][i][j].size();
    }

    return N_i / 2.0 + alpha_tau_i_l;
}

double compute_update_beta_hat_tau(int i, int l, const Parameters &parameters)
{
    double beta_tau_i_l = parameters.beta_tau[i][l];

    double Y_squared_sum = 0.0;
    double Y_xi_sum = 0.0;
    double lambda_xi_sum = 0.0;
    double xi_product_sum = 0.0;

    size_t max_j = parameters.Y[l][i].size();
    size_t max_m = parameters.n_factors;
    for (size_t j = 0; j < max_j; ++j)
    {
        size_t max_k = parameters.Y[l][i][j].size();
        for (size_t k = 0; k < max_k; ++k)
        {
            double Y_ijk_l = parameters.Y[l][i][j][k];
            Y_squared_sum += Y_ijk_l * Y_ijk_l;

            double xi_sum_m = 0.0;
            for (size_t m = 0; m < max_m; ++m)
            {
                double xi_product = xi_L(i, j, k, m, parameters) * xi_F(m, l, parameters);
                double lambda_product = lambda_L(i, j, k, m, parameters) * lambda_F(m, l, parameters);
                xi_sum_m += xi_product;
                lambda_xi_sum += lambda_product - xi_product * xi_product;
            }
            Y_xi_sum += Y_ijk_l * xi_sum_m;
            xi_product_sum += xi_sum_m * xi_sum_m;
        }
    }

    return beta_tau_i_l + 0.5 * (Y_squared_sum - 2 * Y_xi_sum + lambda_xi_sum + xi_product_sum);
}

UpdateTauResult compute_update_tau(int i, int l, const Parameters &parameters)
{
    double update_alpha_hat_tau_i_l = compute_update_alpha_hat_tau(i, l, parameters);
    double update_beta_hat_tau_i_l = compute_update_beta_hat_tau(i, l, parameters);

    return {
        update_alpha_hat_tau_i_l,
        update_beta_hat_tau_i_l};
}

double compute_update_alpha_hat_t(int i, int l, const Parameters &parameters)
{
    double alpha_t_i_l = parameters.alpha_t[i][l];

    double r_sum = 0.0;
    for (size_t j = 0; j < parameters.Y[l][i].size(); ++j)
    {
        for (size_t k = 0; k < parameters.Y[l][i][j].size(); ++k)
        {
            r_sum += parameters.r_L[l][i][j][k];
        }
    }

    return r_sum / 2.0 + alpha_t_i_l;
}

double compute_update_beta_hat_t(int i, int l, const Parameters &parameters)
{
    double beta_t_i_l = parameters.beta_t[i][l];

    double lambda_sum = 0.0;
    for (size_t j = 0; j < parameters.Y[l][i].size(); ++j)
    {
        for (size_t k = 0; k < parameters.Y[l][i][j].size(); ++k)
        {
            lambda_sum += lambda_L(i, j, k, l, parameters);
        }
    }
    return beta_t_i_l + 0.5 * lambda_sum;
}

UpdateTResult compute_update_t(int i, int l, const Parameters &parameters)
{
    double update_alpha_hat_t_i_l = compute_update_alpha_hat_t(i, l, parameters);
    double update_beta_hat_t_i_l = compute_update_beta_hat_t(i, l, parameters);

    return {
        update_alpha_hat_t_i_l,
        update_beta_hat_t_i_l};
}