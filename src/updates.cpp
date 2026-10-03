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

    // Fast BLAS Level 2 matrix-vector product dot_Y = Y_mat * nu_vec
    Eigen::Map<const Eigen::VectorXd> nu_vec(ctx_i_l.nu_L.data(), parameters.n_features);
    Eigen::VectorXd dot_Y_flat = parameters.Y_mats[i] * nu_vec;

    ctx_i_l.dot_Y.resize(parameters.mu_L[l][i].size());
    for (size_t j = 0; j < ctx_i_l.dot_Y.size(); ++j)
    {
        ctx_i_l.dot_Y[j].resize(parameters.mu_L[l][i][j].size());
        for (size_t k = 0; k < ctx_i_l.dot_Y[j].size(); ++k)
        {
            int p = parameters.res_maps[i].jk_to_p[j][k];
            ctx_i_l.dot_Y[j][k] = dot_Y_flat(p);
        }
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



SharedProjections compute_shared_projections(const Parameters &parameters)
{
    int n_res = parameters.n_resolutions;
    int n_factors = parameters.n_factors;
    int n_features = parameters.n_features;

    SharedProjections sp;
    sp.lambda_bar_L.assign(n_res, std::vector<double>(n_factors, 0.0));
    sp.phi_L_res.assign(n_res, std::vector<std::vector<double>>(n_factors, std::vector<double>(n_factors, 0.0)));
    sp.nu_L_res.assign(n_res, std::vector<std::vector<double>>(n_factors, std::vector<double>(n_features, 0.0)));

    for (int i = 0; i < n_res; ++i)
    {
        int N_i = parameters.res_maps[i].N_i;
        Eigen::MatrixXd Xi_L_mat(N_i, n_factors);

        for (int m = 0; m < n_factors; ++m)
        {
            double sum_lam = 0.0;
            for (size_t j = 0; j < parameters.mu_L[m][i].size(); ++j)
            {
                for (size_t k = 0; k < parameters.mu_L[m][i][j].size(); ++k)
                {
                    int p = parameters.res_maps[i].jk_to_p[j][k];
                    Xi_L_mat(p, m) = xi_L(i, j, k, m, parameters);
                    sum_lam += lambda_L(i, j, k, m, parameters);
                }
            }
            sp.lambda_bar_L[i][m] = sum_lam;
        }

        // Fast BLAS Gram matrix: phi_mat = Xi_L_mat^T * Xi_L_mat (K x K)
        Eigen::MatrixXd phi_mat = Xi_L_mat.transpose() * Xi_L_mat;
        for (int m = 0; m < n_factors; ++m)
        {
            sp.phi_L_res[i][m][m] = 0.0;
            for (int mp = 0; mp < n_factors; ++mp)
            {
                if (m != mp)
                {
                    sp.phi_L_res[i][m][mp] = phi_mat(m, mp);
                }
            }
        }

        // Fast BLAS Data projection: nu_mat = Xi_L_mat^T * Y_mats[i] (K x G)
        Eigen::MatrixXd nu_mat = Xi_L_mat.transpose() * parameters.Y_mats[i];
        for (int m = 0; m < n_factors; ++m)
        {
            for (int g = 0; g < n_features; ++g)
            {
                sp.nu_L_res[i][m][g] = nu_mat(m, g);
            }
        }
    }

    return sp;
}

double compute_update_sigma_squared_F(int i, int j, const Parameters &parameters)
{
    double u_bar_L_i_j = u_bar_L(i, j, parameters);
    return 1.0 / (1 + u_bar_L_i_j);
}

UpdateFZFResult compute_update_F_Z_F(int i, int j, const SharedProjections &sp, const Parameters &parameters)
{
    double update_sigma_squared_F_i_j = compute_update_sigma_squared_F(i, j, parameters);
    double s_bar_L_i_j = s_bar_L(i, j, sp, parameters);
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
double compute_update_alpha_hat_tau(int i, int l, const Parameters &parameters)
{
    double alpha_tau_i_l = parameters.alpha_tau[i][l];
    double N_i = parameters.N_coefs_per_res[i];
    return N_i / 2.0 + alpha_tau_i_l;
}

double compute_update_beta_hat_tau(int i, int l, const SharedProjections &sp, double precomputed_Y_squared_sum, const Parameters &parameters)
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

        Y_xi_sum += xi_F_m_l * sp.nu_L_res[i][m][l];
        lambda_sum += lambda_F_m_l * sp.lambda_bar_L[i][m];

        double inner_phi = 0.0;
        for (int mp = 0; mp < n_factors; ++mp)
        {
            inner_phi += sp.phi_L_res[i][m][mp] * xi_F(mp, l, parameters);
        }
        phi_sum += xi_F_m_l * inner_phi;
    }

    return beta_tau_i_l + 0.5 * (precomputed_Y_squared_sum - 2.0 * Y_xi_sum + lambda_sum + phi_sum);
}

UpdateTauResult compute_update_tau(int i, int l, const SharedProjections &sp, double precomputed_Y_squared_sum, const Parameters &parameters)
{
    double update_alpha_hat_tau_i_l = compute_update_alpha_hat_tau(i, l, parameters);
    double update_beta_hat_tau_i_l = compute_update_beta_hat_tau(i, l, sp, precomputed_Y_squared_sum, parameters);

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