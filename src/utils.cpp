#include <RcppEigen.h>
#include <math.h>
#include "penalty.hpp"
// [[Rcpp::depends(RcppEigen)]]

Eigen::VectorXd Compute_Pi_Vec(const Eigen::MatrixXd &x_mat,
                               const Eigen::VectorXd &delta,
                               const Eigen::VectorXd &eta_stack){
/*
 *  This function computes the pi(probability) vector for the logistic regression
 *  Args: x_mat: covariate matrix, size = n * (h + kn * p)
 *        delta: coefficient vector for non-functional covariates
 *               delta.size() = h
 *        eta_stack: coefficient vector for expressed funcitonal covariates
 *                   eta_stack.size() = kn * p
 *                   There are p functional covariates, each one has dimension kn
 *  Return: the pi(probability) vector:
 *          logit = x_mat * coef
 *          pi = exp(logit) / (1 + exp(logit))
 */
    const int h = delta.size();
    const int pkn = eta_stack.size();
    Eigen::VectorXd coef = Eigen::MatrixXd::Zero(h + pkn, 1);
    Eigen::VectorXd logit_vec;
    Eigen::VectorXd pi_vec;
    coef.block(0, 0, h, 1) = delta;
    coef.block(h, 0, pkn, 1) = eta_stack;
    logit_vec = x_mat * coef;

    // a save-guard for logit value
    for(int i = 0; i < logit_vec.size(); i++){
        if(logit_vec[i] > 500){
            logit_vec[i] = 500;
        }else{
            if(logit_vec[i] <= -500){
                logit_vec[i] = -500;
            }
        }
    }
    pi_vec = exp(logit_vec.array()) / (1 + exp(logit_vec.array()));
    return(pi_vec);
}



Eigen::VectorXd Rowsum_wo_j(const Eigen::VectorXd &eta_stack, const int &j,
                            const int &kn, const int &p){
/*
 * Compute the row sum of eta_mat(coming from eta_stack) without one given column j
 * The eta_stack is stored as: e_{11}, ..., e_{1kn}, e_{21}, ..., e_{2kn}, ..., e_{pkn}
 * There are p functional covariates and each one has dimension kn.
 * The eta_mat has size kn * p:
 *     e_{11} , ... , e_{p1}
 *     ...    , ... , ...
 *     e_{1kn}, ... , e_{pkn}
 * This function will compute the row sum of eta_mat, while skipping one column indicated by j.
 * Args: eta_stack: coefficent vector for functional covariates, stacked in one row.
 *       j: index of column of eta_mat. The corresponding column will be skipped when computing row sum
 *       kn: dimension of each functional covariate
 *       p: number of functional covariates
 *
 */
    Eigen::VectorXd rowsum = Eigen::MatrixXd::Zero(kn, 1);
    int start_idx;
    int i;
    for(i = 0; i < j; i++){
        start_idx = i * kn;
        rowsum = rowsum + eta_stack.block(start_idx, 0, kn, 1);
    }
    for(i = j + 1; i < p; i++){
        start_idx = i * kn;
        rowsum = rowsum + eta_stack.block(start_idx, 0, kn, 1);
    }
    return(rowsum);
}



Eigen::VectorXd Rowsum(const Eigen::VectorXd &eta_stack, const int &kn, const int &p){
/*
 * A row sum function for eta_mat.
 * The eta_stack is stored as: e_{11}, ..., e_{1kn}, e_{21}, ..., e_{2kn}, ..., e_{pkn}
 * There are p functional covariates and each one has dimension kn.
 * The eta_mat has size kn * p:
 *     e_{11} , ... , e_{p1}
 *     ...    , ... , ...
 *     e_{1kn}, ... , e_{pkn}
 * This function will compute the row sum of eta_mat.
 * Args: eta_stack: coefficent vector for functional covariates, stacked in one row.
 *       kn: dimension of each functional covariate
 *       p: number of functional covariates
 */
    Eigen::VectorXd rowsum = Eigen::MatrixXd::Zero(kn, 1);
    int start_idx;
    int i;
    for(i = 0; i < p; i++){
        start_idx = i * kn;
        rowsum = rowsum + eta_stack.block(start_idx, 0, kn, 1);
    }
    return(rowsum);
}

double Compute_Loss_Cpp(const Eigen::MatrixXd &x_mat, const Eigen::VectorXd &y_vec,
                    const Eigen::VectorXd &delta_vec, const Eigen::VectorXd &eta_stack_vec,
                    const Eigen::VectorXd &mu1_vec, const double &mu2,
                    const double &h, const double &kn, const double &p,
                    const char &p_type, const Eigen::VectorXd &p_param,
                    const double &a, const Eigen::VectorXd &bj_vec, const Eigen::VectorXd &cj_vec, const Eigen::VectorXd &rj_vec, const Eigen::VectorXd &weight_vec, 
                    const bool &oracle_loss, const bool &print_res){
/*
 *  This function computes the objective value(loss value).
 *  The objective function is consisted of
 *    1. -loglik
 *    2. penalty
 *    3. ADMM regulizer
 *  For oracle estimator, there is no penalty term.
 *
 */
    const int n = y_vec.size();
    Eigen::VectorXd coef = Eigen::MatrixXd::Zero(h + p * kn, 1);
    Eigen::VectorXd logit_vec;
    double loglik, loss_p0, loss_p1, loss_p2, loss_p3, loss;
    Eigen::VectorXd eta_rowsum = Rowsum(eta_stack_vec, kn, p);
    int start_idx, stop_idx;
    Eigen::VectorXd eta_vec;

    double (*pfun)(const double &, const Eigen::VectorXd &, const bool &);

    // loss part 0, the -loglik
    coef.block(0, 0, h, 1) = delta_vec;
    coef.block(h, 0, p * kn, 1) = eta_stack_vec;
    logit_vec = x_mat * coef;
    loglik = ((y_vec.array() * logit_vec.array() - log(1 + exp(logit_vec.array()))) * weight_vec.array()).sum();
    loglik = loglik / a;
    loss_p0 = -loglik;

    // loss part2, the ADMM term
    loss_p2 = mu1_vec.dot(eta_rowsum) + mu2 / 2.0 * eta_rowsum.dot(eta_rowsum);

    // loss part3, the ridge regularizer
    loss_p3 = 0.0;
    for(int i = 0; i < h; i++){
        loss_p3 = loss_p3 + rj_vec[i] / 2.0 * delta_vec[i] * delta_vec[i];
    }
    for(int i = 0; i < p; i++){
        start_idx = i * kn;
        // stop_idx = i * kn + h + (kn - 1);
        eta_vec = eta_stack_vec.block(start_idx, 0, kn, 1);
        loss_p3 = loss_p3 + rj_vec[h + i] / 2.0 * eta_vec.dot(eta_vec);
    }

    loss = loss_p0 + loss_p2 + loss_p3;
    if(oracle_loss){
        if(print_res){
            Rcpp::Rcout << "1 / a * loglik = " << loglik << ", loss_p2 = " << loss_p2 << ", loss_p3 = "<< loss_p3 << ", loss = " << loss << std::endl;
        }
    }else{
        // determine penalty function
        if(p_type == 'L'){
            pfun = Penalty_Lasso;
        }else if(p_type == 'S'){
            pfun = Penalty_SCAD;
        }else if(p_type == 'M'){
            pfun = Penalty_MCP;
        }else{
            Rcpp::Rcout << "Not found!" << std::endl;
            pfun = Penalty_Lasso;
        }

        loss_p1 = 0;
        for(int i = 0; i < p; i++){
            start_idx = i * kn;
            eta_vec = eta_stack_vec.block(start_idx, 0, kn, 1);
            loss_p1 = loss_p1 + cj_vec[i] * pfun(eta_vec.norm() * bj_vec[i], p_param, false);
        }
        loss = loss + loss_p1;
        if(print_res){
            Rcpp::Rcout << "1 / a * loglik = " << loglik << ", loss_p1 = " << loss_p1 <<", loss_p2 = " << loss_p2 << ", loss_p3 = "<< loss_p3 << ", loss = " << loss << std::endl;
        }
    }
    return(loss);
}




