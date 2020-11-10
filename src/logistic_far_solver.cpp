#include<RcppEigen.h>
#include<math.h>

// [[Rcpp::depends(RcppEigen)]]

/*
 * This is the core script of Logistic_FAR_Solver
 */

// pre-define functions
double SoftThresholding(const double z, const double r) {
/*
     This function is used to perform softthresholding on z at r
     res = z - r    if z > fabs(r)
           z + r    if z < -fabs(r)
           0        o.w.
     Args: z: input value.
           r: thresholding point, use its absolute value.
     Returns: res: softhresholding of z at r
*/
    double holding_point;
    holding_point = fabs(r);
    if(z > holding_point) {
        return(z - holding_point);
    }
    if(z < -holding_point) {
        return(z + holding_point);
    }
    return(0.0);
}


double Penalty_Lasso(const double &input, 
                     const Eigen::VectorXd &param, 
                     const bool &derivative){
/*
 * This is Lasso penalty on one double input.
 * Args: input: input value
 *       param: lambda = param[0]
 *       derivative: should the derivative be returned instead of the penalized value.
 */
    const double lam = fabs(param[0]);
    double res;
    if(derivative){
        if(input >= 0){
            res = lam;
        }else{
            res = -lam;
        }
    }else{
        res = lam * fabs(input);
    }
    return(res);
}


double Penalty_SCAD(const double &input, 
                    const Eigen::VectorXd &param, 
                    const bool &derivative){
/*
 * This is SCAD penalty on one double input.
 * Args: input: input value
 *       param: lambda = param[0]
 *              gamma = param[1]
 *       derivative: should the derivative be returned instead of the penalized value.
 */
    const double lam = fabs(param[0]);
    const double gam = fabs(param[1]);
    double res;
    double tmp;
    if(derivative){
            tmp= input;
            if(tmp > lam * gam){
                res = 0;
            }else if(tmp >= lam){
                res = (lam * gam - tmp) / (gam - 1);
            }else if(tmp >= 0){
                res = lam;
            }else if(tmp > -lam){
                res = -lam;
            }else if(tmp > -lam * gam){
                res = (-gam * lam - tmp) / (gam - 1);
            }else{
                res = 0;
            }
    }else{
            tmp = input;
            if(tmp> lam * gam){
                res = (gam + 1) * lam * lam / 2;
            }else if(tmp >= lam){
                res = (-tmp * tmp / 2 + gam * lam * tmp) / (gam - 1) - lam * lam / 2 / (gam - 1);
            }else if(tmp > -lam){
                res = lam * fabs(tmp);
            }else if(tmp > -lam * gam){
                res = (-tmp * tmp / 2 - gam * lam * tmp) / (gam - 1) - lam * lam / 2 / (gam - 1);
            }else{
                res = (gam + 1) * lam * lam / 2;
            }
    }
    return(res);
}


double Penalty_MCP(const double &input, 
                   const Eigen::VectorXd &param, 
                   const bool &derivative){
    /*
     * This is MCP penalty on one double input.
     * Args: input: input value
     *       param: lambda = param[0]
     *              gamma = param[1]
     *       derivative: should the derivative be returned instead of the penalized value.
     */
    const double lam = fabs(param[0]);
    const double gam = fabs(param[1]);
    double res;
    double tmp;
    if(derivative){
            tmp= input;
            if(tmp > gam * lam){
                res = 0;
            }else if(tmp >= 0){
                res = lam - tmp / gam;
            }else if(tmp > -lam * gam){
                res = -lam - tmp / gam;
            }else{
                res = 0;
            }
    }else{
            tmp = input;
            if(tmp > gam * lam){
                res = gam * lam * lam / 2;
            }else if(tmp >= 0){
                res = lam * tmp - tmp * tmp / 2 / gam;
            }else if(tmp > -gam * lam){
                res = -lam * tmp - tmp * tmp / 2 / gam;   
            }else{
                res = gam * lam * lam / 2;
            }
    }
    return(res);
}


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
                    const double &a, const Eigen::VectorXd &bj_vec, 
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
    double loglik, loss_p0, loss_p1, loss_p2, loss;
    Eigen::VectorXd eta_rowsum = Rowsum(eta_stack_vec, kn, p);
    int start_idx;
    Eigen::VectorXd eta_vec;
    
    double (*pfun)(const double &, const Eigen::VectorXd &, const bool &);
    // loss part 0, the -loglik
    coef.block(0, 0, h, 1) = delta_vec;
    coef.block(h, 0, p * kn, 1) = eta_stack_vec;
    logit_vec = x_mat * coef;
    loglik = (y_vec.array() * logit_vec.array() - log(1 + exp(logit_vec.array()))).sum();
    loglik = loglik / a;
    loss_p0 = -loglik;
    
    // loss part2, the ADMM term
    loss_p2 = mu1_vec.dot(eta_rowsum) + mu2 / 2.0 * eta_rowsum.dot(eta_rowsum);
    
    loss = loss_p0 + loss_p2;
    if(oracle_loss){
        if(print_res){
            Rcpp::Rcout << "1 / a * loglik = " << loglik << ", loss_p2 = " << loss_p2 << ", loss = " << loss << std::endl;
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
            loss_p1 = loss_p1 + pfun(eta_vec.norm() * bj_vec[i], 
                                     p_param, false);
        }
        loss = loss + loss_p1;
        if(print_res){
            Rcpp::Rcout << "1 / a * loglik = " << loglik << ", loss_p1 = " << loss_p1 <<", loss_p2 = " << loss_p2 << ", loss = " << loss << std::endl;
        }   
    }
    return(loss);
}

// [[Rcpp::export]]
Rcpp::List Logistic_FAR_Solver_Core(const Eigen::VectorXd &y_vec, const Eigen::MatrixXd &x_mat, 
                                    const int &h, const int &kn, const int &p, 
                                    const char &p_type, const Eigen::VectorXd &p_param, 
                                    const double &mu2, 
                                    const double &a, const Eigen::VectorXd &bj_vec, 
                                    const double &tol, const int &max_iter, 
                                    const Eigen::MatrixXd &h_inv, const Eigen::VectorXd &relax_vec, 
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
                        mu2, h, kn, p, p_type, p_param, a, bj_vec, false, true);
    while((!converge) && (current_iter <= max_iter) && loss_drop){
        // store the results from last iteration
        delta_old = delta;
        eta_stack_old = eta_stack;
        mu1_vec_old = mu1_vec;
        loss_old = loss;
        
        // step1. get the current pi_vec
        pi_vec = Compute_Pi_Vec(x_mat, delta_old, eta_stack_old);
        
        // step2. update demographical covariates
        delta = delta_old - h_inv * (delta_mat.transpose()) * (pi_vec - y_vec);
        pi_vec = Compute_Pi_Vec(x_mat, delta, eta_stack_old);  // get current pi vector
        
        // step3. update the functional covariates
        for(int j = 0; j < p; j++){
            stack_start_idx = j * kn;
            stack_stop_idx = (j + 1) * kn - 1;
            eta_j_old = eta_stack_old.block(stack_start_idx, 0, kn, 1);
            eta_sum_wo_j = Rowsum_wo_j(eta_stack, j, kn, p);
            theta_j = x_mat.block(0, stack_start_idx + h, n, kn);
            alpha_j = (relax_vec[j] - mu2) * eta_j_old + 1.0 / a * (theta_j.transpose()) * (y_vec - pi_vec) - mu1_vec_old - mu2 * eta_sum_wo_j;
            lambda = pfun(sqrt(eta_j_old.dot(eta_j_old)) * bj_vec[j], 
                          p_param, true);
            lambda = bj_vec[j] * lambda;
            positive_check = 1 - lambda / sqrt(alpha_j.dot(alpha_j));
            if(positive_check <= 0){
                eta_stack.block(stack_start_idx, 0, kn, 1) = Eigen::MatrixXd::Zero(kn, 1);
            }else{
                eta_stack.block(stack_start_idx, 0, kn, 1) = 1.0 / relax_vec[j] * positive_check * alpha_j;
            }
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
                                mu2, h, kn, p, p_type, p_param, a, bj_vec, false, false);
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
                            mu2, h, kn, p, p_type, p_param, a, bj_vec, false, true);
    res = Rcpp::List::create(
        Rcpp::Named("delta", delta), 
        Rcpp::Named("eta_stack", eta_stack), 
        Rcpp::Named("mu_1_vec", mu1_vec), 
        Rcpp::Named("iter_num", current_iter), 
        Rcpp::Named("converge", converge), 
        Rcpp::Named("loss_drop", loss_drop)
    );
    return(res);
}

//[[Rcpp::export]]
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

/*
 ******------ Functions for the witnin-group orthonormalization implementation ------******
 */
Eigen::VectorXd Rowsum_Ortho(const Eigen::VectorXd &eta_stack_vec, 
                             const Eigen::MatrixXd &t_mat_stack, 
                             const Eigen::VectorXd &start_id_vec){
    /*
     * 
     */
    const int kn = t_mat_stack.rows();    // length of the original basis expression vector
    const int p = start_id_vec.size() - 1;    // number of functional covariates
    int start_idx = 0; 
    int stop_idx = 0;
    Eigen::VectorXd eta_vec;
    Eigen::MatrixXd t_mat;
    Eigen::VectorXd res = Eigen::MatrixXd::Zero(kn, 1);
    res = t_mat_stack * eta_stack_vec;
    // for(int i = 0; i < p; i++){
    //     start_idx = start_id_vec[i];
    //     stop_idx = start_id_vec[i + 1] - 1;
    //     
    //     eta_vec = eta_stack_vec.block(start_idx, 0, stop_idx - start_idx + 1, 1);
    //     t_mat = t_mat_stack.block(0, start_idx, kn, stop_idx - start_idx + 1);
    //     res = res + t_mat * eta_vec;
    // }
    return(res);
}

Eigen::VectorXd Rowsum_Ortho_wo_j(const Eigen::VectorXd &eta_stack_vec, 
                                  const Eigen::MatrixXd &t_mat_stack, 
                                  const Eigen::VectorXd &start_id_vec, 
                                  const int &j){
    /*
     * 
     */
    const int kn = t_mat_stack.rows();    // length of the original basis expression vector
    const int p = start_id_vec.size() - 1;    // number of functional covariates
    int start_idx = 0; 
    int stop_idx = 0;
    Eigen::VectorXd eta_vec;
    Eigen::MatrixXd t_mat;
    Eigen::VectorXd res = Eigen::MatrixXd::Zero(kn, 1);
    for(int i = 0; i < j; i++){
        start_idx = start_id_vec[i];
        stop_idx = start_id_vec[i + 1] - 1;
        
        eta_vec = eta_stack_vec.block(start_idx, 0, stop_idx - start_idx + 1, 1);
        t_mat = t_mat_stack.block(0, start_idx, kn, stop_idx - start_idx + 1);
        res = res + t_mat * eta_vec;
    }
    for(int i = j + 1; i < p; i++){
        start_idx = start_id_vec[i];
        stop_idx = start_id_vec[i + 1] - 1;
        
        eta_vec = eta_stack_vec.block(start_idx, 0, stop_idx - start_idx + 1, 1);
        t_mat = t_mat_stack.block(0, start_idx, kn, stop_idx - start_idx + 1);
        res = res + t_mat * eta_vec;
    }
    return(res);
}

double Compute_Loss_Ortho_Cpp(const Eigen::MatrixXd &x_mat, const Eigen::VectorXd &y_vec, 
                              const Eigen::VectorXd &delta_vec, const Eigen::VectorXd &eta_stack_vec, 
                              const Eigen::VectorXd &mu1_vec, const double &mu2, 
                              const double &h, const double &kn, const double &p, 
                              const char &p_type, const Eigen::VectorXd &p_param, 
                              const double &a, const Eigen::VectorXd &bj_vec, const Eigen::VectorXd &cj_vec, const Eigen::VectorXd &rj_vec, 
                              const Eigen::MatrixXd &t_mat_stack, 
                              const Eigen::VectorXd &start_id_vec, 
                              const bool &oracle_loss, const bool &print_res){
    /*
     *  This function computes the objective value(loss value).
     *  The objective function is consisted of
     *    1. -loglik
     *    2. penalty 
     *    3. ADMM regulizer
     *    4. ridge-like regularizer
     *  For oracle estimator, there is no penalty term.
     *
     */
    const int n = y_vec.size();
    const int len = t_mat_stack.cols();    // number of grouped variables
                                           // t_mat_stack.cols() = eta_stack_vec.size() = x_mat.cols() - h
    Eigen::VectorXd coef = Eigen::MatrixXd::Zero(h + len, 1);
    Eigen::VectorXd logit_vec;
    double loglik, loss_p0, loss_p1, loss_p2, loss_p3, loss;
    
    Eigen::VectorXd eta_rowsum = t_mat_stack * eta_stack_vec; 
    
    int start_idx, stop_idx;
    Eigen::VectorXd eta_vec;
    
    double (*pfun)(const double &, const Eigen::VectorXd &, const bool &);
    
    // loss part 0, the -loglik
    coef.block(0, 0, h, 1) = delta_vec;
    coef.block(h, 0, len, 1) = eta_stack_vec;
    logit_vec = x_mat * coef;
    loglik = (y_vec.array() * logit_vec.array() - log(1 + exp(logit_vec.array()))).sum();
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
        start_idx = start_id_vec[i];
        stop_idx = start_id_vec[i + 1] - 1;
        eta_vec = eta_stack_vec.block(start_idx, 0, stop_idx - start_idx + 1, 1);
        loss_p3 = loss_p3 + rj_vec[i + h] / 2.0 * eta_vec.dot(eta_vec);
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
            start_idx = start_id_vec[i];
            stop_idx = start_id_vec[i + 1] - 1;
            eta_vec = eta_stack_vec.block(start_idx, 0, stop_idx - start_idx + 1, 1);
            loss_p1 = loss_p1 + cj_vec[i] * pfun(eta_vec.norm() * bj_vec[i], p_param, false);
        }
        loss = loss + loss_p1;
        if(print_res){
            Rcpp::Rcout << "1 / a * loglik = " << loglik << ", loss_p1 = " << loss_p1 <<", loss_p2 = " << loss_p2 << ", loss_p3 = "<< loss_p3 << ", loss = " << loss << std::endl;
        }   
    }
    return(loss);
}


// [[Rcpp::export]]
Rcpp::List Logistic_FAR_Ortho_Solver_Core(const Eigen::VectorXd &y_vec, const Eigen::MatrixXd &x_mat, 
                                          const int &h, const int &kn, const int &p, 
                                          const char &p_type, const Eigen::VectorXd &p_param, 
                                          const double &mu2, 
                                          const double &a, const Eigen::VectorXd &bj_vec, const Eigen::VectorXd &cj_vec, const Eigen::VectorXd &rj_vec, 
                                          const double &tol, const int &max_iter, 
                                          const Eigen::VectorXd &relax_vec, 
                                          const Eigen::MatrixXd &t_mat_stack, 
                                          const Eigen::VectorXd &svd_vec_stack, 
                                          const Eigen::VectorXd &start_id_vec, 
                                          const Eigen::MatrixXd &hd_mat, 
                                          const Eigen::MatrixXd &hd_inv, 
                                          const Eigen::VectorXd &delta_init, 
                                          const Eigen::VectorXd &eta_stack_init, 
                                          const Eigen::VectorXd &mu1_init){
    /*
     * This is the core part of logistic FAR orthogonal solver
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
    Eigen::MatrixXd theta_j;    // covariate matrix of orthogonalized functional part
    Eigen::VectorXd alpha_j;
    Eigen::VectorXd dj_vec;    // singular value vector for the j-th functional part
    Eigen::MatrixXd tj_mat;    // transformation matrix for the j-th functional part
    
    double (*pfun)(const double &, const Eigen::VectorXd &, const bool &);
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
    Rcpp::Rcout << "Before the algorithm:" << std::endl;
    loss = Compute_Loss_Ortho_Cpp(x_mat, y_vec, delta, eta_stack, mu1_vec, 
                                  mu2, h, kn, p, p_type, p_param, a, bj_vec, cj_vec, rj_vec, 
                                  t_mat_stack, start_id_vec, 
                                  false, true);
    while((!converge) && (current_iter < max_iter) && loss_drop){
        // store the results from last iteration
        delta_old = delta;
        eta_stack_old = eta_stack;
        mu1_vec_old = mu1_vec;
        loss_old = loss;
        
        // step1. get the current pi_vec
        pi_vec = Compute_Pi_Vec(x_mat, delta_old, eta_stack_old);
        
        // step2. update demographical covariates
        // delta = delta_old - h_inv * (delta_mat.transpose()) * (pi_vec - y_vec);
        // Rcpp::Rcout << "before delta" << delta[0] << std::endl;
        delta = hd_inv * (hd_mat * delta_old - (delta_mat.transpose()) * (pi_vec - y_vec));
        // Rcpp::Rcout << "after delta" << delta[0] << std::endl;
        pi_vec = Compute_Pi_Vec(x_mat, delta, eta_stack_old);  // get current pi vector
        
        // step3. update the functional covariates
        // Rcpp::Rcout << "before eta" << eta_stack.transpose() << std::endl;
        for(int j = 0; j < p; j++){
            stack_start_idx = start_id_vec[j];
            stack_stop_idx = start_id_vec[j + 1] - 1;
            eta_j_old = eta_stack_old.block(stack_start_idx, 0, stack_stop_idx - stack_start_idx + 1, 1);
            eta_sum_wo_j = Rowsum_Ortho_wo_j(eta_stack, t_mat_stack, start_id_vec, j);
            theta_j = x_mat.block(0, stack_start_idx + h, n, stack_stop_idx - stack_start_idx + 1);
            
            dj_vec = svd_vec_stack.block(stack_start_idx, 0, stack_stop_idx - stack_start_idx + 1, 1);
            tj_mat = t_mat_stack.block(0, stack_start_idx, kn, stack_stop_idx - stack_start_idx + 1);
            
            dj_vec = relax_vec[j] - rj_vec[j] - a * mu2 * dj_vec.array().pow(-2);
            alpha_j = dj_vec.cwiseProduct(eta_j_old) + 1.0 / a * (theta_j.transpose()) * (y_vec - pi_vec) - tj_mat.transpose() * mu1_vec_old - mu2 * tj_mat.transpose() * eta_sum_wo_j;
            
            lambda = pfun(sqrt(eta_j_old.dot(eta_j_old)) * bj_vec[j], 
                          p_param, true);
            lambda = cj_vec[j] * bj_vec[j] * lambda;
            positive_check = 1 - lambda / sqrt(alpha_j.dot(alpha_j));
            if(positive_check <= 0){
                eta_stack.block(stack_start_idx, 0, stack_stop_idx - stack_start_idx + 1, 1) = Eigen::MatrixXd::Zero(stack_stop_idx - stack_start_idx + 1, 1);
            }else{
                eta_stack.block(stack_start_idx, 0, stack_stop_idx - stack_start_idx + 1, 1) = 1.0 / relax_vec[j] * positive_check * alpha_j;
            }
            pi_vec = Compute_Pi_Vec(x_mat, delta, eta_stack);
        }
        // Rcpp::Rcout << "after eta" << eta_stack.transpose() << std::endl;
        // step4. Update mu1
        mu1_vec = mu1_vec_old + mu2 * t_mat_stack * eta_stack;
        
        // check convergency
        current_iter = current_iter + 1;
        diff1 = (delta- delta_old).norm();
        diff2 = (eta_stack - eta_stack_old).norm();
        diff = std::max(diff1, diff2);
        
        loss = Compute_Loss_Ortho_Cpp(x_mat, y_vec, delta, eta_stack, mu1_vec, 
                                      mu2, h, kn, p, p_type, p_param, a, bj_vec, cj_vec, rj_vec, 
                                      t_mat_stack, start_id_vec, 
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
    loss = Compute_Loss_Ortho_Cpp(x_mat, y_vec, delta, eta_stack, mu1_vec, 
                                  mu2, h, kn, p, p_type, p_param, a, bj_vec, cj_vec, rj_vec, 
                                  t_mat_stack, start_id_vec, 
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
