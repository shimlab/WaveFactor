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


// For F_i_j, Z_F_i_j related updates
FZFUpdateContext make_F_Z_F_update_context(int i, const Parameters &parameters)
{
    FZFUpdateContext ctx_i;

    // 1. Spatial wavelet data projection nu_F[a][g] = sum_{b, c} xi_L(a, b, c, i) * Y[g][a][b][c]
    ctx_i.nu_F.assign(parameters.n_resolutions, std::vector<double>(parameters.n_features, std::numeric_limits<double>::signaling_NaN()));
    for (int a = 0; a < parameters.n_resolutions; ++a)
    {
        for (int g = 0; g < parameters.n_features; ++g)
        {
            double sum_xi_Y = 0.0;
            for (std::size_t b = 0; b < parameters.mu_L[i][a].size(); ++b)
            {
                for (std::size_t c = 0; c < parameters.mu_L[i][a][b].size(); ++c)
                {
                    sum_xi_Y += xi_L(a, b, c, i, parameters) * parameters.Y[g][a][b][c];
                }
            }
            ctx_i.nu_F[a][g] = sum_xi_Y;
        }
    }

    // 2. Spatial factor Gram matrix phi_F[a][m] = sum_{b, c} xi_L(a, b, c, i) * xi_L(a, b, c, m)
    ctx_i.phi_F.assign(parameters.n_resolutions, std::vector<double>(parameters.n_factors, std::numeric_limits<double>::signaling_NaN()));
    for (int a = 0; a < parameters.n_resolutions; ++a)
    {
        for (int m = 0; m < parameters.n_factors; ++m)
        {
            if (m == i)
            {
                ctx_i.phi_F[a][m] = 0.0;
                continue;
            }
            double sum_xi_cross = 0.0;
            for (std::size_t b = 0; b < parameters.mu_L[i][a].size(); ++b)
            {
                for (std::size_t c = 0; c < parameters.mu_L[i][a][b].size(); ++c)
                {
                    sum_xi_cross += xi_L(a, b, c, i, parameters) * xi_L(a, b, c, m, parameters);
                }
            }
            ctx_i.phi_F[a][m] = sum_xi_cross;
        }
    }

    return ctx_i;
}

double compute_update_sigma_squared_F(int i, int j, const Parameters &parameters)
{
    double u_bar_L_i_j = u_bar_L(i, j, parameters);
    return 1.0 / (1 + u_bar_L_i_j);
}

UpdateFZFResult compute_update_F_Z_F(int i, int j, const FZFUpdateContext &ctx_i, const Parameters &parameters)
{
    double update_sigma_squared_F_i_j = compute_update_sigma_squared_F(i, j, parameters);
    double s_bar_L_i_j = s_bar_L(i, j, ctx_i, parameters);
    double update_mu_F_i_j = s_bar_L_i_j * update_sigma_squared_F_i_j;

    double log_pi_F_i_j = parameters.log_pi_F[i][j];
    double update_log_r_F_i_j;
    if (log_pi_F_i_j < EFFECTIVE_LOG_ZERO)
    {
        update_log_r_F_i_j = EFFECTIVE_LOG_ZERO;
    }
    else if (log_pi_F_i_j > EFFECTIVE_LOG_ONE)
    {
        update_log_r_F_i_j = EFFECTIVE_LOG_ONE;
    }
    else
    {
        double pi_F_i_j = std::exp(log_pi_F_i_j);
        double log_scaling_factor = 0.5 * (std::log(2.0) + std::log(M_PI) + std::log(update_sigma_squared_F_i_j));
        double log_exp_factor = 0.5 * (-std::log(2.0 * M_PI) + update_mu_F_i_j * update_mu_F_i_j / update_sigma_squared_F_i_j);
        double relative_true_log_prob = log_scaling_factor + log_exp_factor + log_pi_F_i_j;
        double relative_false_log_prob = std::log(1.0 - pi_F_i_j);

        double true_log_prob = relative_true_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);
        double false_log_prob = relative_false_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);

        double incremented_true_log_prob = sum_log(true_log_prob, LOG_RELATIVE_PMF_INCREMENT);
        double incremented_false_log_prob = sum_log(false_log_prob, LOG_RELATIVE_PMF_INCREMENT);

        update_log_r_F_i_j = incremented_true_log_prob - sum_log(incremented_true_log_prob, incremented_false_log_prob);
    }

    return {
        update_sigma_squared_F_i_j,
        update_mu_F_i_j,
        update_log_r_F_i_j};
}

// For tau_i_l related updates
TauUpdateContext make_tau_update_context(int i, const Parameters &parameters)
{
    TauUpdateContext ctx_i;

    // 1. Resolution energy sum lambda_bar_L[m] = sum_{j, k} lambda_L(i, j, k, m)
    ctx_i.lambda_bar_L.assign(parameters.n_factors, std::numeric_limits<double>::signaling_NaN());
    for (int m = 0; m < parameters.n_factors; ++m)
    {
        double sum_lam = 0.0;
        for (std::size_t j = 0; j < parameters.mu_L[m][i].size(); ++j)
        {
            for (std::size_t k = 0; k < parameters.mu_L[m][i][j].size(); ++k)
            {
                sum_lam += lambda_L(i, j, k, m, parameters);
            }
        }
        ctx_i.lambda_bar_L[m] = sum_lam;
    }

    // 2. Spatial factor Gram matrix phi_tau[m][mp] = sum_{j, k} xi_L(i, j, k, m) * xi_L(i, j, k, mp) with diagonal zeroed
    ctx_i.phi_tau.assign(parameters.n_factors, std::vector<double>(parameters.n_factors, std::numeric_limits<double>::signaling_NaN()));
    for (int m = 0; m < parameters.n_factors; ++m)
    {
        ctx_i.phi_tau[m][m] = 0.0;
        for (int mp = m + 1; mp < parameters.n_factors; ++mp)
        {
            double sum_xi_cross = 0.0;
            for (std::size_t j = 0; j < parameters.mu_L[m][i].size(); ++j)
            {
                for (std::size_t k = 0; k < parameters.mu_L[m][i][j].size(); ++k)
                {
                    sum_xi_cross += xi_L(i, j, k, m, parameters) * xi_L(i, j, k, mp, parameters);
                }
            }
            ctx_i.phi_tau[m][mp] = sum_xi_cross;
            ctx_i.phi_tau[mp][m] = sum_xi_cross;
        }
    }

    // 3. Spatial wavelet data projection nu_tau[m][g] = sum_{j, k} Y[g][i][j][k] * xi_L(i, j, k, m)
    ctx_i.nu_tau.assign(parameters.n_factors, std::vector<double>(parameters.n_features, std::numeric_limits<double>::signaling_NaN()));
    for (int m = 0; m < parameters.n_factors; ++m)
    {
        for (int g = 0; g < parameters.n_features; ++g)
        {
            double sum_xi_Y = 0.0;
            for (std::size_t j = 0; j < parameters.mu_L[m][i].size(); ++j)
            {
                for (std::size_t k = 0; k < parameters.mu_L[m][i][j].size(); ++k)
                {
                    sum_xi_Y += parameters.Y[g][i][j][k] * xi_L(i, j, k, m, parameters);
                }
            }
            ctx_i.nu_tau[m][g] = sum_xi_Y;
        }
    }

    return ctx_i;
}

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

double compute_update_beta_hat_tau(int i, int l, const TauUpdateContext &ctx_i, double precomputed_Y_squared_sum, const Parameters &parameters)
{
    double beta_tau_i_l = parameters.beta_tau[i][l];

    double Y_xi_sum = 0.0;
    double lambda_sum = 0.0;
    double phi_sum = 0.0;

    int n_factors = parameters.n_factors;
    for (int m = 0; m < n_factors; ++m)
    {
        double xi_F_m_l = xi_F(m, l, parameters);
        double lambda_F_m_l = lambda_F(m, l, parameters);

        Y_xi_sum += xi_F_m_l * ctx_i.nu_tau[m][l];
        lambda_sum += lambda_F_m_l * ctx_i.lambda_bar_L[m];

        double inner_phi = 0.0;
        for (int mp = 0; mp < n_factors; ++mp)
        {
            inner_phi += ctx_i.phi_tau[m][mp] * xi_F(mp, l, parameters);
        }
        phi_sum += xi_F_m_l * inner_phi;
    }

    return beta_tau_i_l + 0.5 * (precomputed_Y_squared_sum - 2.0 * Y_xi_sum + lambda_sum + phi_sum);
}

UpdateTauResult compute_update_tau(int i, int l, const TauUpdateContext &ctx_i, double precomputed_Y_squared_sum, const Parameters &parameters)
{
    double update_alpha_hat_tau_i_l = compute_update_alpha_hat_tau(i, l, parameters);
    double update_beta_hat_tau_i_l = compute_update_beta_hat_tau(i, l, ctx_i, precomputed_Y_squared_sum, parameters);

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