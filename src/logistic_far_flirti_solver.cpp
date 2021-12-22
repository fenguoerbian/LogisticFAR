#include <RcppEigen.h>
#include <math.h>
#include "penalty.hpp"
#include "utils.hpp"
// [[Rcpp::depends(RcppEigen)]]

/*
 * This is the core script of Logistic_FAR_FLiRTI_Solver
 */

// pre-define functions

// [[Rcpp::export]]
Rcpp::List Logistic_FAR_FLiRTI_Solver_Core(const Eigen::VectorXd &y_vec, const Eigen::MatrixXd &x_mat,
                                           const int &h, const int &kn, const int &p,
                                           const char &p_type, const Eigen::VectorXd &p_param,
                                           const double &mu2,
                                           const double &a, const Eigen::VectorXd &bj_vec, const Eigen::VectorXd &cj_vec, const Eigen::VectorXd &rj_vec,
                                           const double &tol, const int &max_iter,
                                           const Eigen::VectorXd &relax_vec,
                                           const Eigen::MatrixXd &hd_mat,
                                           const Eigen::MatrixXd &hd_inv,
                                           const Eigen::VectorXd &delta_init,
                                           const Eigen::VectorXd &eta_stack_init,
                                           const Eigen::VectorXd &mu1_init){
/*
 * This is the core part of logistic FAR solver
 * Args: y_vec: response vector, 0 or 1.
 *       x_mat: covariate matrix, n * (h + kn * p) matrix
 *              first h columns are demographical covariates
 *              then kn * p columns are functional covariates
 *       h: integer, number of demographical covariates
 *       kn: integer, dimension of expression for functional covariates
 *       p: integer, number of functional covariates
 *       p_type: character, penalty type
 *               'L': Lasso
 *               'S': SCAD
 *               'M': MCP
 *       p_param: numerical vector storing parameters for penalty function
 *       mu2: double, parameter for quadratic term in ADMM algorithm
 *       a: double, parameter for likelihood part
 *       bj_vec: numerical vector for parameters in penalty kernel
 *       tol: double tolerence for converge
 *       max_iter: integer, number of max iteration
 *
 */
    const int n = y_vec.size();    // number of observations(subjects)
    double diff = 1;
    double diff1, diff2;
    int current_iter = 0;
    bool converge = false;
    bool loss_drop = true;
    double loss;
    double loss_old;
    int stack_start_idx, stack_stop_idx;
    double lambda;
    double positive_check;
    Rcpp::List res;

    // some variables during iteration
    Eigen::VectorXd delta = delta_init;
    Eigen::VectorXd eta_stack = eta_stack_init;
    Eigen::VectorXd mu1_vec = mu1_init;
    Eigen::VectorXd delta_old = delta;
    Eigen::VectorXd eta_stack_old = eta_stack;
    Eigen::VectorXd mu1_vec_old = mu1_vec;
    // intermediate variables
    Eigen::VectorXd pi_vec;    // pi vector
    const Eigen::MatrixXd delta_mat = x_mat.block(0, 0, n, h);    // covariate matrix of demographical part
    Eigen::VectorXd eta_j_old;
    Eigen::VectorXd eta_sum_wo_j;
    Eigen::MatrixXd theta_j;
    Eigen::VectorXd alpha_j;

    // Rcpp::Function r_compute_loss("Compute_Loss");    // access R function Compute_Loss
    double (*pfun)(const double &, const Eigen::VectorXd &, const bool &);
    // determine penalty function
    if(p_type == 'L'){
        pfun = Penalty_Lasso;
    }else if(p_type == 'S'){
        pfun = Penalty_SCAD;
    }else if(p_type == 'M'){
        pfun = Penalty_MCP;
    }else{
        Rcpp::Rcout << "Penalty not found! Use default Lasso!" << std::endl;
        pfun = Penalty_Lasso;
    }
    Rcpp::Rcout << "Before the algorithm:" << std::endl;
    /*
    loss = Rcpp::as<double>(r_compute_loss(Rcpp::Named("x_mat", x_mat),
                                           Rcpp::Named("y_vec", y_vec),
                                           Rcpp::Named("delta_vec", delta),
                                           Rcpp::Named("eta_stack_vec", eta_stack),
                                           Rcpp::Named("mu1_vec", mu1_vec_old),
                                           Rcpp::Named("mu_2", mu2),
                                           Rcpp::Named("h", h),
                                           Rcpp::Named("kn", kn),
                                           Rcpp::Named("p", p),
                                           Rcpp::Named("p_type", p_type),
                                           Rcpp::Named("p_param", p_param),
                                           Rcpp::Named("a", a),
                                           Rcpp::Named("bj_vec", bj_vec),
                                           Rcpp::Named("oracle_loss", false),
                                           Rcpp::Named("print_res", true)));
     */
    loss = Compute_Loss_Cpp(x_mat, y_vec, delta, eta_stack, mu1_vec,
                            mu2, h, kn, p, p_type, p_param, a, bj_vec, cj_vec, rj_vec,
                            false, true);

    while((!converge) && (current_iter <= max_iter) && loss_drop){
        // store the results from last iteration
        delta_old = delta;
        eta_stack_old = eta_stack;
        mu1_vec_old = mu1_vec;
        loss_old = loss;

        // step1. get the current pi_vec
        pi_vec = Compute_Pi_Vec(x_mat, delta_old, eta_stack_old);

        // step2. update demographical covariates
        // delta = delta_old - h_inv * (delta_mat.transpose()) * (pi_vec - y_vec);
        delta = hd_inv * (hd_mat * delta_old - (delta_mat.transpose()) * (pi_vec - y_vec));
        pi_vec = Compute_Pi_Vec(x_mat, delta, eta_stack_old);  // get current pi vector

        // step3. update the functional covariates
        for(int j = 0; j < p; j++){
            stack_start_idx = j * kn;
            stack_stop_idx = (j + 1) * kn - 1;
            eta_j_old = eta_stack_old.block(stack_start_idx, 0, kn, 1);
            eta_sum_wo_j = Rowsum_wo_j(eta_stack, j, kn, p);
            theta_j = x_mat.block(0, stack_start_idx + h, n, kn);

            alpha_j = (relax_vec[j] - mu2 - rj_vec[j + h]) * eta_j_old + 1.0 / a * (theta_j.transpose()) * (y_vec - pi_vec) - mu1_vec_old - mu2 * eta_sum_wo_j;
            
            for(int k = 0; k < kn; k++){
                lambda = pfun(eta_j_old[k] * bj_vec[j], 
                              p_param, true);
                lambda = cj_vec[j] * bj_vec[j] * lambda;
                eta_stack.block(stack_start_idx + k, 0, 1, 1) = (alpha_j[k], lambda) / relax_vec[j];
            }
            // lambda = pfun(sqrt(eta_j_old.dot(eta_j_old)) * bj_vec[j],
            //               p_param, true);
            // lambda = cj_vec[j] * bj_vec[j] * lambda;
            // positive_check = 1 - lambda / sqrt(alpha_j.dot(alpha_j));
            // if(positive_check <= 0){
            //     eta_stack.block(stack_start_idx, 0, kn, 1) = Eigen::MatrixXd::Zero(kn, 1);
            // }else{
            //     eta_stack.block(stack_start_idx, 0, kn, 1) = 1.0 / relax_vec[j] * positive_check * alpha_j;
            // }
            pi_vec = Compute_Pi_Vec(x_mat, delta, eta_stack);
        }

        // step4. Update mu1
        mu1_vec = mu1_vec_old + mu2 * Rowsum(eta_stack, kn, p);

        // check convergency
        current_iter = current_iter + 1;
        diff1 = (delta- delta_old).norm();
        diff2 = (eta_stack - eta_stack_old).norm();
        diff = std::max(diff1, diff2);
        /*
        loss = Rcpp::as<double>(r_compute_loss(Rcpp::Named("x_mat", x_mat),
                                               Rcpp::Named("y_vec", y_vec),
                                               Rcpp::Named("delta_vec", delta),
                                               Rcpp::Named("eta_stack_vec", eta_stack),
                                               Rcpp::Named("mu1_vec", mu1_vec),
                                               Rcpp::Named("mu_2", mu2),
                                               Rcpp::Named("h", h),
                                               Rcpp::Named("kn", kn),
                                               Rcpp::Named("p", p),
                                               Rcpp::Named("p_type", p_type),
                                               Rcpp::Named("p_param", p_param),
                                               Rcpp::Named("a", a),
                                               Rcpp::Named("bj_vec", bj_vec),
                                               Rcpp::Named("oracle_loss", false),
                                               Rcpp::Named("print_res", false)));
         */
        loss = Compute_Loss_Cpp(x_mat, y_vec, delta, eta_stack, mu1_vec,
                            mu2, h, kn, p, p_type, p_param, a, bj_vec, cj_vec, rj_vec,
                            false, false);
        if(loss > loss_old + 3){
            loss_drop = false;
            if(diff <= tol){
                converge = true;
            }
        }else{
            loss_drop = true;
            if(diff <= tol){
                converge = true;
            }
        }
    }
    Rcpp::Rcout << "iter_num = " << current_iter << ", diff1 = " << diff1 << ", diff2 = " << diff2 << ", loss = " << loss << std::endl;
    Rcpp::Rcout << "after the algorithm" << std::endl;
    // Rcpp::Rcout << delta << std::endl;
    /*
    loss = Rcpp::as<double>(r_compute_loss(Rcpp::Named("x_mat", x_mat),
                                           Rcpp::Named("y_vec", y_vec),
                                           Rcpp::Named("delta_vec", delta),
                                           Rcpp::Named("eta_stack_vec", eta_stack),
                                           Rcpp::Named("mu1_vec", mu1_vec),
                                           Rcpp::Named("mu_2", mu2),
                                           Rcpp::Named("h", h),
                                           Rcpp::Named("kn", kn),
                                           Rcpp::Named("p", p),
                                           Rcpp::Named("p_type", p_type),
                                           Rcpp::Named("p_param", p_param),
                                           Rcpp::Named("a", a),
                                           Rcpp::Named("bj_vec", bj_vec),
                                           Rcpp::Named("oracle_loss", false),
                                           Rcpp::Named("print_res", true)));
     */
    loss = Compute_Loss_Cpp(x_mat, y_vec, delta, eta_stack, mu1_vec,
                            mu2, h, kn, p, p_type, p_param, a, bj_vec, cj_vec, rj_vec,
                            false, true);
                            
    res = Rcpp::List::create(
        Rcpp::Named("delta", delta),
        Rcpp::Named("eta_stack", eta_stack),
        Rcpp::Named("mu1_vec", mu1_vec),
        Rcpp::Named("iter_num", current_iter),
        Rcpp::Named("converge", converge),
        Rcpp::Named("loss_drop", loss_drop)
    );
    return(res);
}


/* This function is not compatible with the current Solver_Core function
Rcpp::List Logistic_FAR_Path_Core(const Eigen::VectorXd &y_vec, const Eigen::MatrixXd &x_mat,
                                  const int &h, const int &kn, const int &p,
                                  const char &p_type, Eigen::VectorXd &p_param,
                                  const double &mu2,
                                  const double &a, const Eigen::VectorXd &bj_vec,
                                  const double &tol, const int &max_iter,
                                  const Eigen::MatrixXd &h_inv, const Eigen::VectorXd &relax_vec,
                                  const Eigen::VectorXd &delta_init,
                                  const Eigen::VectorXd &eta_stack_init,
                                  const Eigen::VectorXd &mu1_init,
                                  const Eigen::VectorXd &lambda_seq){
    const int n = y_vec.size();
    const int lam_len = lambda_seq.size();
    Eigen::MatrixXd delta_path = Eigen::MatrixXd::Zero(lam_len, h);
    Eigen::MatrixXd eta_stack_path = Eigen::MatrixXd::Zero(lam_len, p * kn);
    Eigen::MatrixXd mu1_path = Eigen::MatrixXd::Zero(lam_len, kn);
    Eigen::VectorXi iter_num_path = Eigen::MatrixXi::Zero(lam_len, 1);
    Eigen::VectorXi converge_path = Eigen::MatrixXi::Zero(lam_len, 1);
    Eigen::VectorXi loss_drop_path = Eigen::MatrixXi::Zero(lam_len, 1);

    Eigen::VectorXd delta_old = delta_init;
    Eigen::VectorXd eta_stack_old = eta_stack_init;
    Eigen::VectorXd mu1_old = mu1_init;

    Rcpp::List far_res;
    Rcpp::List far_path_res;
    for(int i = 0; i < lam_len; i++){
        p_param[0] = lambda_seq[i];
        far_res = Logistic_FAR_Solver_Core(y_vec, x_mat, h, kn, p, p_type, p_param,
                                           mu2, a, bj_vec, tol, max_iter, h_inv, relax_vec,
                                           delta_old, eta_stack_old, mu1_old);
        delta_path.row(i) = Rcpp::as<Eigen::VectorXd>(far_res["delta"]);
        eta_stack_path.row(i) = Rcpp::as<Eigen::VectorXd>(far_res["eta_stack"]);
        mu1_path.row(i) = Rcpp::as<Eigen::VectorXd>(far_res["mu_1_vec"]);
        iter_num_path[i] = Rcpp::as<int>(far_res["iter_num"]);
        converge_path[i] = Rcpp::as<int>(far_res["converge"]);
        loss_drop_path[i] = Rcpp::as<int>(far_res["loss_drop"]);

        delta_old = Rcpp::as<Eigen::VectorXd>(far_res["delta"]);
        eta_stack_old = Rcpp::as<Eigen::VectorXd>(far_res["eta_stack"]);
        mu1_old = Rcpp::as<Eigen::VectorXd>(far_res["mu_1_vec"]);

        Rcpp::Rcout << "Lambda ID = " << i << ", lambda = " << lambda_seq[i] << " finished!" << std::endl;
    }
    far_path_res = Rcpp::List::create(
        Rcpp::Named("delta_path", delta_path),
        Rcpp::Named("eta_stack_path", eta_stack_path),
        Rcpp::Named("mu_1_path", mu1_path),
        Rcpp::Named("iter_num_path", iter_num_path),
        Rcpp::Named("converge_path", converge_path),
        Rcpp::Named("loss_drop_path", loss_drop_path),
        Rcpp::Named("lambda_seq", lambda_seq)
    );
    return(far_path_res);
}

*/
