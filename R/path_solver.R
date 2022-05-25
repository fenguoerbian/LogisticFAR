#' Finds the solution path of logistic functional additive regression with log-contrast constrain.
#'
#' \code{Logistic_FAR_Path} finds the solution path of logistic functional additive
#' regression with log-contrast constrain. It will NOT perform within-group orthonormalization
#' as preprocession of the data.
#'
#' @param y_vec response vector, 0 for control, 1 for case.
#' n = length(y_vec) is the number of observations.
#'
#' @param x_mat covariate matrix, consists of two parts.
#' dim(x_mat) = (n, h + p * kn)
#' First h columns are for demographical covariates(can include an intercept term)
#' Rest columns are for p functional covariates, each being represented by a set of basis functions resulting kn covariates.
#'
#' @param h,kn,p dimension information for the dataset(\code{x_mat}).
#'
#' @param p_type an character variable indicating different types of the penalty
#               "L": lasso;
#               "S": SCAD;
#               "M": MCP
#'
#' @param p_param numerical vector for the penalty function.
#' \code{p_param[1]} store sthe lambda value and will be provided by \code{lambda_seq}.
#'
#' @param lambda_seq a non-negative sequence of lambda, along which the solution path is searched.
#' It is RECOMMENED to not supply this parameter and let the function itself determines
#' it from the given data.
#'
#' @param lambda_length length of the lambda sequence when computing \code{lambda_seq}.
#' If \code{lambda_seq} is provided, then of course \code{lambda_length = length(lambda_seq)}.
#'
#' @param min_lam_ratio: \code{min(lambda_seq) / max{lambda_seq}}. This function uses this
#' parameter to determine the minimal value of \code{lambda_seq}. If \code{p > n}, then it
#' is recommended to set this no smaller than 0.01 (sometimes even 0.05), otherwise you can
#' set it to 0.001 or even smaller.
#'
#' @param mu2 quadratic term in the ADMM algorithm
#'
#' @param a,bj_vec,cj_vec,rj_vec parameters for the algorithm. See Algorithm_Details.pdf
#' for more information.
#'
#' @param weight_vec weight vector for each subject.
#'   The final weight for each subject will be adjusted also by \code{logit_weight_vec}.
#'   And the summation of the final weight vector is normalized to \code{n}, the sample size.
#'
#' @param logit_weight_vec weight vector for each subject when computing the integral in the logit values.
#'   Each entry should be positive and no more than 1.
#'   This is a naive method for adjusting for early stop during the interval.
#'
#' @param weight_already_combine boolen, indicating whether the \code{weight_vec}
#'   is already combined with \code{logit_weight_vec} for each subject.
#'
#' @param delta_init,eta_stack_init,mu1_init initial values for the algorithm.
#'
#' @param tol,max_iter convergence tolerance and max number of iteration of the algorithm.
#'
#' @param verbose not used
#'
#' @param svd_thresh not used
#'
#' @return A list containing the solution path of \code{delta}, \code{eta_stack}, \code{mu1}
#' and some computation information such as convergency, iteration number and the lambda
#' sequence of this solution path.
#' @export
Logistic_FAR_Path <- function(y_vec, x_mat, h, kn, p,
                              p_type, p_param,
                              lambda_seq, lambda_length, min_lambda_ratio = 0.01,
                              mu2, a = 1, bj_vec = 1, cj_vec = sqrt(kn), rj_vec = 10^(-6),
                              weight_vec = 1, logit_weight_vec = 1, weight_already_combine = FALSE,
                              delta_init, eta_stack_init, mu1_init,
                              tol = 10 ^ (-6), max_iter = 500, verbose = TRUE, svd_thresh = 10^{-7}){
    # This function finds the solution path of Logistic_FAR over a sequence of lambda
    # Note: x_mat is the basis coefficient representation version,
    #         NOT the original functional version
    # Args: p_type: an character variable indicating different types of the penalty
    #               "L": lasso;
    #               "S": SCAD;
    #               "M": MCP
    #       p_param: numerical parameters for the penalty function
    #                p_param[1] is always the lambda
    #                In this solution path function, p_param[1] will be provided by
    #                lambda_seq
    #       a: numerical scalar, the 1st part of loss function is 1 / a * (-loglik)
    #       bj_vec: numerical vector for the penalty kernel
    #               the penalty term is penalty_{lambda}(bj * \|eta\|)
    #
    # Return:

    ######------------ prepare the data ------------
    y_vec <- as.vector(y_vec)
    x_mat <- as.matrix(x_mat)
    n <- length(y_vec)    # number of observations
    if(n != nrow(x_mat)){
        stop("x_mat and y_vec don't have the same number of observations")
    }
    if((h + kn * p) != ncol(x_mat)){
        stop("supplied h, kn or p don't match with column number of x_mat!")
    }

    ###--- check a, bj, cj and rj_vec ---###
    if(length(bj_vec) == 1){
        bj_vec <- rep(bj_vec, p)
    }else{
        if(length(bj_vec) != p){
            stop("length of bj_vec does not match p!")
        }
    }
    if(length(cj_vec) == 1){
        cj_vec <- rep(cj_vec, p)
    }else{
        if(length(cj_vec) != p){
            stop("length of cj_vec does not match p!")
        }
    }
    if(length(rj_vec) == 1){
        rj_vec <- rep(rj_vec, p + h)
    }else{
        if(length(rj_vec) != (p + h)){
            stop("length of rj_vec does not match (p + h)!")
        }
    }

    ### --- check weight_vec --- ###
    if(any(is.na(weight_vec))){
        message("`weight_vec` contains `NULL`! Set `weight_vec` to 1!")
        weight_vec <- 1
    }
    if(!all(weight_vec > 0)){
        stop("weight_vec must be positive!")
    }

    if(length(weight_vec) == 1){
        weight_vec <- rep(weight_vec, n)
    }else{
        if(length(weight_vec) != n){
            stop("length of weight_vec does not match n!")
        }
    }

    ### --- check logit_weight_vec --- ###
    if(any(is.na(logit_weight_vec))){
        message("`logit_weight_vec` contains `NULL`! Set `logit_weight_vec` to 1!")
        logit_weight_vec <- 1
    }
    if(!all(logit_weight_vec > 0)){
        stop("logit_weight_vec must be positive!")
    }

    if(!all(logit_weight_vec <= 1)){
        stop("logit_weight_vec must be no greater than 1!")
    }

    if(length(logit_weight_vec) == 1){
        logit_weight_vec <- rep(logit_weight_vec, n)
    }else{
        if(length(logit_weight_vec) != n){
            stop("length of logit_weight_vec does not match n!")
        }
    }

    ### --- normalize weight_vec ---
    if(!weight_already_combine){
        weight_vec <-weight_vec * logit_weight_vec
        weight_already_combine <- TRUE
    }

    weight_vec <- weight_vec / sum(weight_vec) * n
    weight_diag_mat <- diag(x = weight_vec, nrow = n)

    # ------ This algorithm do not use within-group orthonormalization ------
    # # standardize those grouped covariates in x_mat
    # x_mat_bak <- x_mat    # a back up of x_mat
    # # transformation matrix, stacked in row
    # t_mat_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
    # for(i in 1 : p){
    #     start_idx <- 1 + h + (i - 1) * k_n
    #     stop_idx <- k_n + h + (i - 1) * k_n
    #     svd_res <- svd(x_mat[, start_idx : stop_idx, drop = FALSE], nu = 0)
    #     t_mat <- sqrt(a) * svd_res$v %*% diag(1 / svd_res$d, nrow = k_n)
    #     t_mat_stack[, (start_idx : stop_idx) - h] <- t_mat
    #     x_mat[, start_idx : stop_idx] <- x_mat[, start_idx : stop_idx] %*% t_mat
    # }

    # ------ covariate matrix for non-functional covariates ------
    delta_mat <- x_mat[, 1 : h, drop = FALSE]
    hd_mat <- 0.25 * t(delta_mat) %*% weight_diag_mat %*% delta_mat
    hd_inv <- solve(hd_mat + diag(a * rj_vec[1 : h], nrow = h))

    # ------ covariate matrices for functional covariates ------
    # ind_mat stores the starting and stopping index for each functional covariates
    #   in x_mat. Each row for one functional covariates.
    # ind_mat also provides starting and stopping indces in eta_stack_vec, just minus h
    #   since there's no delta part in eta.
    ind_mat <- matrix(0, nrow = p, ncol = 2)
    colnames(ind_mat) <- c("start_ind", "stop_ind")
    rownames(ind_mat) <- paste("v", 1 : p, sep = "")
    ind_mat[, 1] <- (0 : (p - 1)) * kn + 1 + h
    ind_mat[, 2] <- (1 : p) * kn + h

    # --- start_id_vec ---
    # start_id_vec, in the same definition of the within-group orthonormalization function.
    start_id_vec <- c(ind_mat[, 1], ind_mat[p, 2] + 1)
    start_id_vec <- start_id_vec - h

    # if(missing(eta_inv_stack)){
    #     eta_inv_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
    #     for(j in 1 : p){
    #         stack_start <- (j - 1) * k_n + 1
    #         stack_stop <- j * k_n
    #         x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
    #         h_mat_j <- 1 / 4 * t(x_mat_j) %*% x_mat_j
    #         eta_inv_stack[, stack_start : stack_stop] <- solve(4 * h_mat_j)
    #     }
    # }

    # --- relax_vec ---
    relax_vec <- rep(1, p)
    for(i in 1 : p){
        start_idx <- ind_mat[i, 1]
        stop_idx <- ind_mat[i, 2]
        x_mat_i <- x_mat[, start_idx : stop_idx, drop = FALSE]
        h_mat_i <- 1 / 4 * t(x_mat_i) %*% weight_diag_mat %*% x_mat_i
        eigen_value_vec <- eigen(h_mat_i, only.values = TRUE)
        eigen_max <- max(eigen_value_vec$values)
        relax_vec[i] <- rj_vec[i + h] + mu2 + eigen_max / a * (1 + 10^(-6))
    }
    rm(x_mat_i)
    rm(h_mat_i)
    rm(eigen_value_vec)
    rm(eigen_max)

    # if(missing(relax_vec)){
    #     relax_vec <- rep(1, p)
    #     # b_eigen_val_vec <- eigen(t(b_mat) %*% b_mat, only.values = TRUE)
    #     # b_eigen_max <- max(b_eigen_val_vec$values)
    #     for(j in 1 : p){
    #         x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
    #         h_mat_j <- 1 / 4 * t(x_mat_j) %*% x_mat_j
    #         eigen_value_vec <- eigen(h_mat_j, only.values = TRUE)
    #         # origin version of relax vector, where penalty kernel is \theta\eta
    #         # eigen_min <- min(eigen_value_vec$values)
    #         # relax_vec[j] <- (1 + 10^(-6)) * (1 + mu_2 * b_eigen_max / eigen_min)
    #         # new version of relax vector, where penalty kernel is \eta
    #         eigen_max <- max(eigen_value_vec$values)
    #         relax_vec[j] <- (1 + 10 ^(-6)) * (mu_2 + eigen_max / a)
    #     }
    #     print("Relax vector is: ")
    #     print(relax_vec)
    # }

    if(missing(lambda_seq)){
        print("lambda sequence is missing, using default method to determine it!")

        if(missing(lambda_length) || missing(min_lambda_ratio)){
            stop("Both lambda_length and min_lambda_ratio must be provided for computing the lambda sequence!")
        }else{
            print(paste("lambda_length = ", lambda_length, sep = ""))
            print(paste("min_lambda_ratio = ", min_lambda_ratio, sep = ""))

            # --- find lambda_max, now in a stand alone function ---
            lam_max <- Get_Lambda_Max(y_vec = y_vec, x_mat = x_mat,
                                      h = h, kn = kn, p = p,
                                      a = a, bj_vec = bj_vec, cj_vec = cj_vec,
                                      start_id_vec = start_id_vec)
            # conduct the ordinary logistic regressoin
            # logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            # pi_fit <- exp(logit_fit$fitted.values) / (1 + exp(logit_fit$fitted.values))
            # alpha_vec <- rep(0, p)
            # for(i in 1 : p){
            #     start_ind <- ind_mat[i, 1]
            #     stop_ind <- ind_mat[i, 2]
            #     theta_i_mat <- x_mat[, start_ind : stop_ind, drop = FALSE]
            #     can_vec <- t(theta_i_mat) %*% (y_vec - pi_fit) / a
            #     # origin version of relax vector, where penalty kernel is \theta\eta
            #     # alpha_vec[i] <- sqrt(n * t(can_vec) %*% solve(t(theta_i_mat) %*% theta_i_mat) %*% can_vec)
            #
            #     # new version of relax vector, where penalty kernel is \eta
            #     alpha_vec[i] <- sqrt(t(can_vec) %*% can_vec) / bj_vec[i]
            # }
            # rm(theta_i_mat)
            # rm(can_vec)
            # lam_max = max(alpha_vec)
            # rm(alpha_vec)

            lam_min = lam_max * min_lambda_ratio
            # lam_min = lam_max
            lambda_seq <- exp(seq(from = log(lam_max), to = log(lam_min), length.out = lambda_length))

            # default initial values for lambda_max
            print("Using default lambda sequences and initial values for the path searching!")
            # delta_init <- rep(0, h)
            logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            delta_init <- logit_fit$coefficients
            eta_stack_init <- rep(0, p * kn)
            # mu_1_init <- rep(0, nrow(b_mat))
            mu1_init <- rep(0, kn)
        }
    }else{
        lambda_length <- length(lambda_seq)
        lambda_seq <- sort(abs(lambda_seq), decreasing = TRUE)

        # check initial values for the algorithm
        if(missing(delta_init)){
            print("delta_init missing, use default settings")
            logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            delta_init <- logit_fit$coefficients
        }else{

        }

        if(missing(eta_stack_init)){
            print("eta_stack_init missing, use default settings")
            eta_stack_init <- rep(0, p * kn)
        }else{
            if(length(eta_stack_init) != p * kn){
                print("length(eta_stack_init) != p * kn. Results might be wrong!")
                eta_stack_init <- eta_stack_init[1 : (p * kn)]
            }
        }

        if(missing(mu1_init)){
            print("mu1_init missing, use default settings")
            mu1_init <- rep(0, kn)
        }
    }

    delta_path <- matrix(0, nrow = lambda_length, ncol = h)
    eta_stack_path <- matrix(0, nrow = lambda_length, ncol = p * kn)
    # mu_1_path <- matrix(0, nrow = lambda_length, ncol = nrow(b_mat))
    mu1_path <- matrix(0, nrow = lambda_length, ncol = kn)
    iter_num_path <- rep(0, lambda_length)
    converge_path <- rep(0, lambda_length)
    loss_drop_path <- rep(0, lambda_length)

    for(lam_ind in 1 : lambda_length){
        # update lambda
        lambda <- lambda_seq[lam_ind]
        p_param[1] <- lambda

        # conduct the algorithm
        FAR_res <- Logistic_FAR_Solver_Core(y_vec = y_vec, x_mat = x_mat, h = h, kn = kn, p = p, p_type = p_type, p_param = p_param,
                                            mu2 = mu2, a = a, bj_vec = bj_vec, cj_vec = cj_vec, rj_vec = rj_vec,
                                            weight_vec = weight_vec, logit_weight_vec = logit_weight_vec,
                                            tol = tol, max_iter = max_iter,
                                            relax_vec = relax_vec, hd_mat = hd_mat, hd_inv = hd_inv,
                                            delta_init = delta_init, eta_stack_init = eta_stack_init, mu1_init = mu1_init)
        # save the result
        delta_path[lam_ind, ] <- FAR_res$delta
        eta_stack_path[lam_ind, ] <- FAR_res$eta_stack
        mu1_path[lam_ind, ] <- FAR_res$mu1_vec
        iter_num_path[lam_ind] <- FAR_res$iter_num
        converge_path[lam_ind] <- FAR_res$converge
        loss_drop_path[lam_ind] <- FAR_res$loss_drop

        # update initial values for the next run
        delta_init <- FAR_res$delta
        eta_stack_init <- FAR_res$eta_stack
        mu1_init <- FAR_res$mu1_vec
        # print some information
        print(paste("Lambda ID = ", lam_ind, ", lambda = ", lambda, " finished!", sep = ""))
    }

    # get the original eta_stack_path
    # for(i in 1 : p){
    #     start_idx <- 1 + (i - 1) * k_n
    #     stop_idx <- k_n + (i - 1) * k_n
    #     t_mat <- t_mat_stack[, start_idx : stop_idx, drop = FALSE]
    #     eta_j_mat <- eta_stack_path[, start_idx : stop_idx, drop = FALSE]
    #     eta_stack_path[, start_idx : stop_idx] <- t(t_mat %*% t(eta_j_mat))
    # }
    # what should we do about the mu_1_path?

    # return the result
    res <- list(delta_path = delta_path,
                eta_stack_path = eta_stack_path,
                mu_1_path = mu1_path,
                iter_num_path = iter_num_path,
                converge_path = converge_path,
                loss_drop_path = loss_drop_path,
                lambda_seq = lambda_seq)
    return(res)
}

#' Cross-validation for solution path of Logistic FAR.
#'
#' \code{Logistic_FAR_CV_path} finds the solution path of logistic functional
#' additive regression with log-contrast constrain via \code{Logistic_FAR_Path}.
#' And it will use cross-validation to assess the goodness of the estimations
#' in the solution path.
#'
#' @param y_vec response vector, 0 for control, 1 for case.
#' n = length(y_vec) is the number of observations.
#'
#' @param x_mat covariate matrix, consists of two parts.
#' dim(x_mat) = (n, h + p * kn)
#' First h columns are for demographical covariates(can include an intercept term)
#' Rest columns are for p functional covariates, each being represented by a set of basis functions resulting kn covariates.
#'
#' @param h,kn,p dimension information for the dataset(\code{x_mat}).
#'
#' @param p_type an character variable indicating different types of the penalty
#               "L": lasso;
#               "S": SCAD;
#               "M": MCP
#'
#' @param p_param numerical vector for the penalty function.
#' \code{p_param[1]} store sthe lambda value and will be provided by \code{lambda_seq}.
#'
#' @param lambda_seq a non-negative sequence of lambda, along which the solution path is searched.
#' It is RECOMMENED to not supply this parameter and let the function itself determines
#' it from the given data.
#'
#' @param lambda_length length of the lambda sequence when computing \code{lambda_seq}.
#' If \code{lambda_seq} is provided, then of course \code{lambda_length = length(lambda_seq)}.
#'
#' @param min_lam_ratio: \code{min(lambda_seq) / max{lambda_seq}}. This function uses this
#' parameter to determine the minimal value of \code{lambda_seq}. If \code{p > n}, then it
#' is recommended to set this no smaller than 0.01 (sometimes even 0.05), otherwise you can
#' set it to 0.001 or even smaller.
#'
#' @param mu2 quadratic term in the ADMM algorithm
#'
#' @param a,bj_vec,cj_vec,rj_vec parameters for the algorithm. See Algorithm_Details.pdf
#' for more information.
#'
#' @param weight_vec weight vector for each subject.
#'   The final weight for each subject will be adjusted also by \code{logit_weight_vec}.
#'   And the summation of the final weight vector is normalized to \code{n}, the sample size.
#'
#' @param logit_weight_vec weight vector for each subject when computing the integral in the logit values.
#'   Each entry should be positive and no more than 1.
#'   This is a naive method for adjusting for early stop during the interval.
#'
#' @param weight_already_combine boolen, indicating whether the \code{weight_vec}
#'   is already combined with \code{logit_weight_vec} for each subject.
#'
#' @param delta_init,eta_stack_init,mu1_init initial values for the algorithm.
#'
#' @param tol,max_iter convergence tolerance and max number of iteration of the algorithm.
#'
#' @param relax_vec not used.
#'
#' @param svd_thresh not used.
#'
#' @param nfold integer, number of folds
#'
#' @param fold_seed if supplied, use this seed to generate the partitions for cross-validation.
#' Can be useful for reproducible runs.
#'
#' @param post_selection bool, should the function also computes cross-validation results
#' based on post selection estimation results.
#'
#' @param post_a \code{a} for the post selection estimation.
#'
#' @return A list containing the solution path of \code{delta}, \code{eta_stack}, \code{mu1}
#' and some computation information such as convergency, iteration number and the lambda
#' sequence of this solution path. Also information of CV is returned such as the fold ID
#' for each observation, the loglikelihood results on each test set and the index with the
#' highest average loglik on the testsets. If \code{post_selection = TRUE}, same results
#' based on the post selection estimation are also returned.
#'
#' @note Although this function will return the index of lambda given the highest
#' averaged loglik on the testsets. It is more recommended to use the stand alone
#' \code{*_pick} functions in this packages, such as \code{CV_Pick} to find a optimal
#' lambda since those functions give more flexibility.
#' @export
Logistic_FAR_CV_path <- function(y_vec, x_mat, h, kn, p,
                                 p_type, p_param,
                                 lambda_seq, lambda_length, min_lambda_ratio = 0.01,
                                 mu2, a = 1, bj_vec = rep(1 / sqrt(kn), p), cj_vec  = rep(1, p), rj_vec = 0.00001,
                                 weight_vec = 1, logit_weight_vec = 1, weight_already_combine = FALSE,
                                 relax_vec,
                                 delta_init, eta_stack_init, mu_1_init,
                                 tol, max_iter, nfold = 5, fold_seed, post_selection = FALSE, post_a = 1){
    # This function finds the solution path of Logistic_FAR over a sequence of lambda
    # It uses cross-validation (based on the largest loglikelihood on the test sets)
    #  to find the best lambda in that lambda sequence
    # Note: x_mat is the basis coefficient representation version,
    #         NOT the original functional version
    # Args: p_type: an character variable indicating different types of the penalty
    #               "L": lasso;
    #               "S": SCAD;
    #               "M": MCP
    #       p_param: numerical parameters for the penalty function
    #                p_param[1] is always the lambda
    #                In this solution path function, p_param[1] will be provided by
    #                lambda_seq
    #       a: numerical scalar, the 1st part of loss function is 1 / a * (-loglik)
    #       bj_vec: numerical vector for the penalty kernel
    #               the penalty term is penalty_{lambda}(bj * \|eta\|)
    #
    # Return:

    ######------------ prepare the data ------------
    y_vec <- as.vector(y_vec)
    x_mat <- as.matrix(x_mat)
    n <- length(y_vec)    # number of observations
    if(n != nrow(x_mat)){
        stop("x_mat and y_vec don't have the same number of observations")
    }
    if((h + kn * p) != ncol(x_mat)){
        stop("supplied h, kn or p don't match with column number of x_mat!")
    }

    ###--- check a, bj, cj and rj_vec ---###
    if(length(bj_vec) == 1){
        bj_vec <- rep(bj_vec, p)
    }else{
        if(length(bj_vec) != p){
            stop("length of bj_vec does not match p!")
        }
    }
    if(length(cj_vec) == 1){
        cj_vec <- rep(cj_vec, p)
    }else{
        if(length(cj_vec) != p){
            stop("length of cj_vec does not match p!")
        }
    }
    if(length(rj_vec) == 1){
        rj_vec <- rep(rj_vec, p + h)
    }else{
        if(length(rj_vec) != (p + h)){
            stop("length of rj_vec does not match (p + h)!")
        }
    }

    ### --- check weight_vec --- ###
    if(any(is.na(weight_vec))){
        message("`weight_vec` contains `NULL`! Set `weight_vec` to 1!")
        weight_vec <- 1
    }
    if(!all(weight_vec > 0)){
        stop("weight_vec must be positive!")
    }

    if(length(weight_vec) == 1){
        weight_vec <- rep(weight_vec, n)
    }else{
        if(length(weight_vec) != n){
            stop("length of weight_vec does not match n!")
        }
    }

    ### --- check logit_weight_vec --- ###
    if(any(is.na(logit_weight_vec))){
        message("`logit_weight_vec` contains `NULL`! Set `logit_weight_vec` to 1!")
        logit_weight_vec <- 1
    }
    if(!all(logit_weight_vec > 0)){
        stop("logit_weight_vec must be positive!")
    }

    if(!all(logit_weight_vec <= 1)){
        stop("logit_weight_vec must be no greater than 1!")
    }

    if(length(logit_weight_vec) == 1){
        logit_weight_vec <- rep(logit_weight_vec, n)
    }else{
        if(length(logit_weight_vec) != n){
            stop("length of logit_weight_vec does not match n!")
        }
    }

    ### --- normalize weight_vec ---
    if(!weight_already_combine){
        weight_vec <-weight_vec * logit_weight_vec
        weight_already_combine <- TRUE
    }

    weight_vec <- weight_vec / sum(weight_vec) * n
    weight_diag_mat <- diag(x = weight_vec, nrow = n)

    # ------ This algorithm do not use within-group orthonormalization ------
    # # standardize those grouped covariates in x_mat
    x_mat_bak <- x_mat    # a back up of x_mat
    # # transformation matrix, stacked in row
    # t_mat_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
    # for(i in 1 : p){
    #     start_idx <- 1 + h + (i - 1) * k_n
    #     stop_idx <- k_n + h + (i - 1) * k_n
    #     svd_res <- svd(x_mat[, start_idx : stop_idx, drop = FALSE], nu = 0)
    #     t_mat <- sqrt(a) * svd_res$v %*% diag(1 / svd_res$d, nrow = k_n)
    #     t_mat_stack[, (start_idx : stop_idx) - h] <- t_mat
    #     x_mat[, start_idx : stop_idx] <- x_mat[, start_idx : stop_idx] %*% t_mat
    # }

    # covariate matrix for non-functional covariates
    # delta_mat <- x_mat[, 1 : h, drop = FALSE]
    # if(missing(h_inv)){
    #   h_mat <- 1 / 4 * t(delta_mat) %*% delta_mat
    #   h_inv <- solve(h_mat)
    # }

    # ------ covariate matrices for functional covariates ------
    # ind_mat stores the starting and stopping index for each functional covariates
    #   in x_mat. Each row for one functional covariates.
    # ind_mat also provides starting and stopping indces in eta_stack_vec, just minus h
    #   since there's no delta part in eta.
    ind_mat <- matrix(0, nrow = p, ncol = 2)
    colnames(ind_mat) <- c("start_ind", "stop_ind")
    rownames(ind_mat) <- paste("v", 1 : p, sep = "")
    ind_mat[, 1] <- (0 : (p - 1)) * kn + 1 + h
    ind_mat[, 2] <- (1 : p) * kn + h

    # --- start_id_vec ---
    # start_id_vec, in the same definition of the within-group orthonormalization function.
    start_id_vec <- c(ind_mat[, 1], ind_mat[p, 2] + 1)
    start_id_vec <- start_id_vec - h

    # if(missing(eta_inv_stack)){
    #     eta_inv_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
    #     for(j in 1 : p){
    #         stack_start <- (j - 1) * k_n + 1
    #         stack_stop <- j * k_n
    #         x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
    #         h_mat_j <- 1 / 4 * t(x_mat_j) %*% x_mat_j
    #         eta_inv_stack[, stack_start : stack_stop] <- solve(4 * h_mat_j)
    #     }
    # }

    if(missing(lambda_seq)){
        print("lambda sequence is missing, using default method to determine it!")

        if(missing(lambda_length) || missing(min_lambda_ratio)){
            stop("Both lambda_length and min_lambda_ratio must be provided for computing the lambda sequence!")
        }else{
            print(paste("lambda_length = ", lambda_length, sep = ""))
            print(paste("min_lambda_ratio = ", min_lambda_ratio, sep = ""))

            # --- find lambda_max, now in a stand alone function ---
            lam_max <- Get_Lambda_Max(y_vec = y_vec, x_mat = x_mat,
                                      h = h, kn = kn, p = p,
                                      a = a, bj_vec = bj_vec, cj_vec = cj_vec,
                                      start_id_vec = start_id_vec)
            # conduct the ordinary logistic regressoin
            # logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            # pi_fit <- exp(logit_fit$fitted.values) / (1 + exp(logit_fit$fitted.values))
            # alpha_vec <- rep(0, p)
            # for(i in 1 : p){
            #     start_ind <- ind_mat[i, 1]
            #     stop_ind <- ind_mat[i, 2]
            #     theta_i_mat <- x_mat[, start_ind : stop_ind, drop = FALSE]
            #     can_vec <- t(theta_i_mat) %*% (y_vec - pi_fit) / a
            #     # origin version of relax vector, where penalty kernel is \theta\eta
            #     # alpha_vec[i] <- sqrt(n * t(can_vec) %*% solve(t(theta_i_mat) %*% theta_i_mat) %*% can_vec)
            #
            #     # new version of relax vector, where penalty kernel is \eta
            #     alpha_vec[i] <- sqrt(t(can_vec) %*% can_vec) / bj_vec[i]
            # }
            # rm(theta_i_mat)
            # rm(can_vec)
            # lam_max = max(alpha_vec)
            # rm(alpha_vec)


            lam_min = lam_max * min_lambda_ratio
            # lam_min = lam_max
            lambda_seq <- exp(seq(from = log(lam_max), to = log(lam_min), length.out = lambda_length))

            # default initial values for lambda_max
            print("Using default lambda sequences and initial values for the path searching!")
            # delta_init <- rep(0, h)
            logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            delta_init <- logit_fit$coefficients
            eta_stack_init <- rep(0, p * kn)
            # mu_1_init <- rep(0, nrow(b_mat))
            mu_1_init <- rep(0, kn)
        }
    }else{
        lambda_length <- length(lambda_seq)
        lambda_seq <- sort(lambda_seq, decreasing = TRUE)

        # check initial values for the algorithm
        if(missing(delta_init)){
            print("delta_init missing, use default settings")
            logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            delta_init <- logit_fit$coefficients
        }else{

        }

        if(missing(eta_stack_init)){
            print("eta_stack_init missing, use default settings")
            eta_stack_init <- rep(0, p * kn)
        }else{
            if(length(eta_stack_init) != p * kn){
                print("length(eta_stack_init) != p * kn. Results might be wrong!")
                eta_stack_init <- eta_stack_init[1 : (p * kn)]
            }
        }

        if(missing(mu_1_init)){
            print("mu1_init missing, use default settings")
            mu_1_init <- rep(0, kn)
        }
    }

    # get fold id
    # if(!missing(fold_seed)){
    #     set.seed(fold_seed)
    # }
    # fold_id_vec <- sample(rep(seq(nfold), length = n))
    fold_id_list <- splitTools::create_folds(as.factor(y_vec),
                                             k = nfold,
                                             type = "stratified",
                                             invert = TRUE,
                                             seed = fold_seed)
    fold_id_vec <- rep(0, length = n)
    for(fold_id in 1 : nfold){
        fold_id_vec[fold_id_list[[fold_id]]] <- fold_id
    }

    # related variables for cv results
    loglik_test_mat <- matrix(0, nrow = nfold, ncol = lambda_length)    # store the loglik on the test set
    # each row for one test set
    if(post_selection){
        loglik_post_mat <- loglik_test_mat
    }

    pb <- progressr::progressor(along = 1 : nfold)
    for(cv_id in 1 : nfold){
        print(paste(nfold, "-fold CV, starting at ", cv_id, "/", nfold, sep = ""))
        test_id_vec <- which(fold_id_vec == cv_id)
        x_mat_train <- x_mat_bak[-test_id_vec, , drop = FALSE]
        y_vec_train <- y_vec[-test_id_vec]
        weight_vec_train <- weight_vec[-test_id_vec]
        logit_weight_vec_train <- logit_weight_vec[-test_id_vec]
        x_mat_test <- x_mat_bak[test_id_vec, , drop = FALSE]
        y_vec_test <- y_vec[test_id_vec]
        weight_vec_test <- weight_vec[test_id_vec]
        logit_weight_vec_test <- logit_weight_vec[test_id_vec]

        # find solution path on the training set
        print(paste("Find solution path on training set..."))
        train_res <- Logistic_FAR_Path(y_vec = y_vec_train, x_mat = x_mat_train,
                                       h = h, kn = kn, p = p, p_type = p_type, p_param = p_param,
                                       lambda_seq = lambda_seq, mu2 = mu2,
                                       a = a, bj_vec = bj_vec, cj_vec = cj_vec, rj_vec = rj_vec,
                                       weight_vec = weight_vec_train,
                                       logit_weight_vec = logit_weight_vec_train,
                                       weight_already_combine = weight_already_combine,
                                       tol = tol, max_iter = max_iter)
        # test performance on the test set
        print(paste("Compute loglik on the testing set..."))
        for(lam_id in 1 : lambda_length){
            delta_vec <- train_res$delta_path[lam_id, ]
            eta_stack_vec <- train_res$eta_stack_path[lam_id, ]
            test_pi_vec <- as.vector((x_mat_test[, 1 : h, drop = FALSE] %*% delta_vec) + (x_mat_test[, -(1 : h), drop = FALSE] %*% eta_stack_vec) * logit_weight_vec_test)
            # test_pi_vec <- as.vector(x_mat_test %*% c(delta_vec, eta_stack_vec))
            loglik_test_mat[cv_id, lam_id] <- sum((y_vec_test * test_pi_vec - log(1 + exp(test_pi_vec))) * weight_vec_test)
        }

        # test on testing set based on post-selection estimation
        if(post_selection){
            # post_res <- train_res
            for(lam_id in 1 : lambda_length){
                post_est <-  Logistic_FAR_Path_Further_Improve(x_mat = x_mat_train, y_vec = y_vec_train, h = h, k_n = kn, p = p,
                                                               delta_vec_init = train_res$delta_path[lam_id, ],
                                                               eta_stack_init = train_res$eta_stack_path[lam_id, ],
                                                               mu1_vec_init = train_res$mu_1_path[lam_id, ],
                                                               # mu1_vec_init = rep(0, k_n),
                                                               mu2 = mu2, a = post_a,
                                                               weight_vec = weight_vec_train,
                                                               logit_weight_vec = logit_weight_vec_train,
                                                               weight_already_combine = weight_already_combine,
                                                               lam = 0.001, tol = 10^{-5}, max_iter = 1000)
                # post_res$delta_path[lam_id, ] <- post_est$delta_vec
                # post_res$eta_stack_path[lam_id, ] <- post_est$eta_stack_vec
                # post_res$mu1_path[lam_id, ] <- post_est$mu1_vec
                # post_res$iter_num_path[lam_id] <- post_est$iter_num
                # post_res$converge_path[lam_id] <- post_est$converge

                delta_vec <- post_est$delta_vec
                peta_stack_vec <- post_est$eta_stack_vec
                test_pi_vec <- as.vector((x_mat_test[, 1 : h, drop = FALSE] %*% delta_vec) + (x_mat_test[, -(1 : h), drop = FALSE] %*% eta_stack_vec) * logit_weight_vec_test)
                # test_pi_vec <- as.vector(x_mat_test %*% c(delta_vec, eta_stack_vec))
                loglik_post_mat[cv_id, lam_id] <- sum((y_vec_test * test_pi_vec - log(1 + exp(test_pi_vec))) * weight_vec_test)
            }

        }
        print(paste(nfold, "-fold CV, FINISHED at ", cv_id, "/", nfold, sep = ""))
        pb(paste(nfold, "-fold CV, folder id = ", cv_id, " finished at pid = ", Sys.getpid(), "!", sep = ""))
    }

    # find the lambda with the highest test loglik
    lam_id <- which.max(colSums(loglik_test_mat))
    res <- Logistic_FAR_Path(y_vec = y_vec, x_mat = x_mat_bak,
                             h = h, kn = kn, p = p, p_type = p_type, p_param = p_param,
                             lambda_seq = lambda_seq, mu2 = mu2,
                             a = a, bj_vec = bj_vec, cj_vec = cj_vec, rj_vec = rj_vec,
                             weight_vec = weight_vec,
                             logit_weight_vec = logit_weight_vec,
                             weight_already_combine = weight_already_combine,
                             tol = tol, max_iter = max_iter)

    res$cv_id <- lam_id
    res$loglik_test_mat <- loglik_test_mat
    res$fold_id_vec <- fold_id_vec

    if(post_selection){
        lam_post_id <- which.max(colSums(loglik_post_mat))
        post_est <- Logistic_FAR_Path_Further_Improve(x_mat = x_mat_bak, y_vec = y_vec, h = h, k_n = kn, p = p,
                                                      delta_vec_init = res$delta_path[lam_post_id, ],
                                                      eta_stack_init = res$eta_stack_path[lam_post_id, ],
                                                      mu1_vec_init = res$mu_1_path[lam_post_id, ],
                                                      # mu1_vec_init = rep(0, k_n),
                                                      mu2 = mu2, a = post_a,
                                                      weight_vec = weight_vec,
                                                      logit_weight_vec = logit_weight_vec,
                                                      weight_already_combine = weight_already_combine,
                                                      lam = 0.001, tol = 10^{-5}, max_iter = 1000)
        res$cv_post_id <- lam_post_id
        res$loglik_post_mat <- loglik_post_mat
        res$post_est <- post_est
    }
    return(res)
}

#' Cross-validation for solution path of Logistic FAR.
#'
#' \code{Logistic_FAR_CV_path_par} finds the solution path of logistic functional
#' additive regression with log-contrast constrain via \code{Logistic_FAR_Path}.
#' And it will use cross-validation to assess the goodness of the estimations
#' in the solution path. The cross-validation is implemented in parallel manner.
#'
#' @param y_vec response vector, 0 for control, 1 for case.
#' n = length(y_vec) is the number of observations.
#'
#' @param x_mat covariate matrix, consists of two parts.
#' dim(x_mat) = (n, h + p * kn)
#' First h columns are for demographical covariates(can include an intercept term)
#' Rest columns are for p functional covariates, each being represented by a set of basis functions resulting kn covariates.
#'
#' @param h,kn,p dimension information for the dataset(\code{x_mat}).
#'
#' @param p_type an character variable indicating different types of the penalty
#               "L": lasso;
#               "S": SCAD;
#               "M": MCP
#'
#' @param p_param numerical vector for the penalty function.
#' \code{p_param[1]} store sthe lambda value and will be provided by \code{lambda_seq}.
#'
#' @param lambda_seq a non-negative sequence of lambda, along which the solution path is searched.
#' It is RECOMMENED to not supply this parameter and let the function itself determines
#' it from the given data.
#'
#' @param lambda_length length of the lambda sequence when computing \code{lambda_seq}.
#' If \code{lambda_seq} is provided, then of course \code{lambda_length = length(lambda_seq)}.
#'
#' @param min_lam_ratio: \code{min(lambda_seq) / max{lambda_seq}}. This function uses this
#' parameter to determine the minimal value of \code{lambda_seq}. If \code{p > n}, then it
#' is recommended to set this no smaller than 0.01 (sometimes even 0.05), otherwise you can
#' set it to 0.001 or even smaller.
#'
#' @param mu2 quadratic term in the ADMM algorithm
#'
#' @param a,bj_vec,cj_vec,rj_vec parameters for the algorithm. See Algorithm_Details.pdf
#' for more information.
#'
#' @param weight_vec weight vector for each subject.
#'   The final weight for each subject will be adjusted also by \code{logit_weight_vec}.
#'   And the summation of the final weight vector is normalized to \code{n}, the sample size.
#'
#' @param logit_weight_vec weight vector for each subject when computing the integral in the logit values.
#'   Each entry should be positive and no more than 1.
#'   This is a naive method for adjusting for early stop during the interval.
#'
#' @param weight_already_combine boolen, indicating whether the \code{weight_vec}
#'   is already combined with \code{logit_weight_vec} for each subject.
#'
#' @param delta_init,eta_stack_init,mu1_init initial values for the algorithm.
#'
#' @param tol,max_iter convergence tolerance and max number of iteration of the algorithm.
#'
#' @param relax_vec not used.
#'
#' @param svd_thresh not used.
#'
#' @param nfold integer, number of folds
#'
#' @param fold_seed if supplied, use this seed to generate the partitions for cross-validation.
#' Can be useful for reproducible runs.
#'
#' @param post_selection bool, should the function also computes cross-validation results
#' based on post selection estimation results.
#'
#' @param post_a \code{a} for the post selection estimation.
#'
#' @return A list containing the solution path of \code{delta}, \code{eta_stack}, \code{mu1}
#' and some computation information such as convergency, iteration number and the lambda
#' sequence of this solution path. Also information of CV is returned such as the fold ID
#' for each observation, the loglikelihood results on each test set and the index with the
#' highest average loglik on the testsets. If \code{post_selection = TRUE}, same results
#' based on the post selection estimation are also returned.
#'
#' @note Although this function will return the index of lambda given the highest
#' averaged loglik on the testsets. It is more recommended to use the stand alone
#' \code{*_pick} functions in this packages, such as \code{CV_Pick} to find a optimal
#' lambda since those functions give more flexibility.
#' @export
Logistic_FAR_CV_path_par <- function(y_vec, x_mat, h, kn, p,
                                     p_type, p_param,
                                     lambda_seq, lambda_length, min_lambda_ratio = 0.01,
                                     mu2, a = 1, bj_vec = rep(1 / sqrt(kn), p), cj_vec  = rep(1, p), rj_vec = 0.00001,
                                     weight_vec = 1, logit_weight_vec = 1, weight_already_combine = FALSE,
                                     relax_vec,
                                     delta_init, eta_stack_init, mu_1_init,
                                     tol, max_iter, nfold = 5, fold_seed, post_selection = FALSE, post_a = 1){
    # This function finds the solution path of Logistic_FAR over a sequence of lambda
    # It uses cross-validation (based on the largest loglikelihood on the test sets)
    #  to find the best lambda in that lambda sequence
    # Note: x_mat is the basis coefficient representation version,
    #         NOT the original functional version
    # Args: p_type: an character variable indicating different types of the penalty
    #               "L": lasso;
    #               "S": SCAD;
    #               "M": MCP
    #       p_param: numerical parameters for the penalty function
    #                p_param[1] is always the lambda
    #                In this solution path function, p_param[1] will be provided by
    #                lambda_seq
    #       a: numerical scalar, the 1st part of loss function is 1 / a * (-loglik)
    #       bj_vec: numerical vector for the penalty kernel
    #               the penalty term is penalty_{lambda}(bj * \|eta\|)
    #
    # Return:

    ######------------ prepare the data ------------
    y_vec <- as.vector(y_vec)
    x_mat <- as.matrix(x_mat)
    n <- length(y_vec)    # number of observations
    if(n != nrow(x_mat)){
        stop("x_mat and y_vec don't have the same number of observations")
    }
    if((h + kn * p) != ncol(x_mat)){
        stop("supplied h, kn or p don't match with column number of x_mat!")
    }

    ###--- check a, bj, cj and rj_vec ---###
    if(length(bj_vec) == 1){
        bj_vec <- rep(bj_vec, p)
    }else{
        if(length(bj_vec) != p){
            stop("length of bj_vec does not match p!")
        }
    }
    if(length(cj_vec) == 1){
        cj_vec <- rep(cj_vec, p)
    }else{
        if(length(cj_vec) != p){
            stop("length of cj_vec does not match p!")
        }
    }
    if(length(rj_vec) == 1){
        rj_vec <- rep(rj_vec, p + h)
    }else{
        if(length(rj_vec) != (p + h)){
            stop("length of rj_vec does not match (p + h)!")
        }
    }

    ### --- check weight_vec --- ###
    if(any(is.na(weight_vec))){
        message("`weight_vec` contains `NULL`! Set `weight_vec` to 1!")
        weight_vec <- 1
    }
    if(!all(weight_vec > 0)){
        stop("weight_vec must be positive!")
    }

    if(length(weight_vec) == 1){
        weight_vec <- rep(weight_vec, n)
    }else{
        if(length(weight_vec) != n){
            stop("length of weight_vec does not match n!")
        }
    }

    ### --- check logit_weight_vec --- ###
    if(any(is.na(logit_weight_vec))){
        message("`logit_weight_vec` contains `NULL`! Set `logit_weight_vec` to 1!")
        logit_weight_vec <- 1
    }
    if(!all(logit_weight_vec > 0)){
        stop("logit_weight_vec must be positive!")
    }

    if(!all(logit_weight_vec <= 1)){
        stop("logit_weight_vec must be no greater than 1!")
    }

    if(length(logit_weight_vec) == 1){
        logit_weight_vec <- rep(logit_weight_vec, n)
    }else{
        if(length(logit_weight_vec) != n){
            stop("length of logit_weight_vec does not match n!")
        }
    }

    ### --- normalize weight_vec ---
    if(!weight_already_combine){
        weight_vec <-weight_vec * logit_weight_vec
        weight_already_combine <- TRUE
    }

    weight_vec <- weight_vec / sum(weight_vec) * n
    weight_diag_mat <- diag(x = weight_vec, nrow = n)

    # ------ This algorithm do not use within-group orthonormalization ------
    # # standardize those grouped covariates in x_mat
    x_mat_bak <- x_mat    # a back up of x_mat
    # # transformation matrix, stacked in row
    # t_mat_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
    # for(i in 1 : p){
    #     start_idx <- 1 + h + (i - 1) * k_n
    #     stop_idx <- k_n + h + (i - 1) * k_n
    #     svd_res <- svd(x_mat[, start_idx : stop_idx, drop = FALSE], nu = 0)
    #     t_mat <- sqrt(a) * svd_res$v %*% diag(1 / svd_res$d, nrow = k_n)
    #     t_mat_stack[, (start_idx : stop_idx) - h] <- t_mat
    #     x_mat[, start_idx : stop_idx] <- x_mat[, start_idx : stop_idx] %*% t_mat
    # }

    # covariate matrix for non-functional covariates
    # delta_mat <- x_mat[, 1 : h, drop = FALSE]
    # if(missing(h_inv)){
    #   h_mat <- 1 / 4 * t(delta_mat) %*% delta_mat
    #   h_inv <- solve(h_mat)
    # }

    # ------ covariate matrices for functional covariates ------
    # ind_mat stores the starting and stopping index for each functional covariates
    #   in x_mat. Each row for one functional covariates.
    # ind_mat also provides starting and stopping indces in eta_stack_vec, just minus h
    #   since there's no delta part in eta.
    ind_mat <- matrix(0, nrow = p, ncol = 2)
    colnames(ind_mat) <- c("start_ind", "stop_ind")
    rownames(ind_mat) <- paste("v", 1 : p, sep = "")
    ind_mat[, 1] <- (0 : (p - 1)) * kn + 1 + h
    ind_mat[, 2] <- (1 : p) * kn + h

    # --- start_id_vec ---
    # start_id_vec, in the same definition of the within-group orthonormalization function.
    start_id_vec <- c(ind_mat[, 1], ind_mat[p, 2] + 1)
    start_id_vec <- start_id_vec - h

    # if(missing(eta_inv_stack)){
    #     eta_inv_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
    #     for(j in 1 : p){
    #         stack_start <- (j - 1) * k_n + 1
    #         stack_stop <- j * k_n
    #         x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
    #         h_mat_j <- 1 / 4 * t(x_mat_j) %*% x_mat_j
    #         eta_inv_stack[, stack_start : stack_stop] <- solve(4 * h_mat_j)
    #     }
    # }

    if(missing(lambda_seq)){
        print("lambda sequence is missing, using default method to determine it!")

        if(missing(lambda_length) || missing(min_lambda_ratio)){
            stop("Both lambda_length and min_lambda_ratio must be provided for computing the lambda sequence!")
        }else{
            print(paste("lambda_length = ", lambda_length, sep = ""))
            print(paste("min_lambda_ratio = ", min_lambda_ratio, sep = ""))

            # --- find lambda_max, now in a stand alone function ---
            lam_max <- Get_Lambda_Max(y_vec = y_vec, x_mat = x_mat,
                                      h = h, kn = kn, p = p,
                                      a = a, bj_vec = bj_vec, cj_vec = cj_vec,
                                      start_id_vec = start_id_vec)
            # conduct the ordinary logistic regressoin
            # logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            # pi_fit <- exp(logit_fit$fitted.values) / (1 + exp(logit_fit$fitted.values))
            # alpha_vec <- rep(0, p)
            # for(i in 1 : p){
            #     start_ind <- ind_mat[i, 1]
            #     stop_ind <- ind_mat[i, 2]
            #     theta_i_mat <- x_mat[, start_ind : stop_ind, drop = FALSE]
            #     can_vec <- t(theta_i_mat) %*% (y_vec - pi_fit) / a
            #     # origin version of relax vector, where penalty kernel is \theta\eta
            #     # alpha_vec[i] <- sqrt(n * t(can_vec) %*% solve(t(theta_i_mat) %*% theta_i_mat) %*% can_vec)
            #
            #     # new version of relax vector, where penalty kernel is \eta
            #     alpha_vec[i] <- sqrt(t(can_vec) %*% can_vec) / bj_vec[i]
            # }
            # rm(theta_i_mat)
            # rm(can_vec)
            # lam_max = max(alpha_vec)
            # rm(alpha_vec)


            lam_min = lam_max * min_lambda_ratio
            # lam_min = lam_max
            lambda_seq <- exp(seq(from = log(lam_max), to = log(lam_min), length.out = lambda_length))

            # default initial values for lambda_max
            print("Using default lambda sequences and initial values for the path searching!")
            # delta_init <- rep(0, h)
            logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            delta_init <- logit_fit$coefficients
            eta_stack_init <- rep(0, p * kn)
            # mu_1_init <- rep(0, nrow(b_mat))
            mu_1_init <- rep(0, kn)
        }
    }else{
        lambda_length <- length(lambda_seq)
        lambda_seq <- sort(lambda_seq, decreasing = TRUE)

        # check initial values for the algorithm
        if(missing(delta_init)){
            print("delta_init missing, use default settings")
            logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            delta_init <- logit_fit$coefficients
        }else{

        }

        if(missing(eta_stack_init)){
            print("eta_stack_init missing, use default settings")
            eta_stack_init <- rep(0, p * kn)
        }else{
            if(length(eta_stack_init) != p * kn){
                print("length(eta_stack_init) != p * kn. Results might be wrong!")
                eta_stack_init <- eta_stack_init[1 : (p * kn)]
            }
        }

        if(missing(mu_1_init)){
            print("mu1_init missing, use default settings")
            mu_1_init <- rep(0, kn)
        }
    }

    # get fold id
    # if(!missing(fold_seed)){
    #     set.seed(fold_seed)
    # }
    # fold_id_vec <- sample(rep(seq(nfold), length = n))
    fold_id_list <- splitTools::create_folds(as.factor(y_vec),
                                             k = nfold,
                                             type = "stratified",
                                             invert = TRUE,
                                             seed = fold_seed)
    fold_id_vec <- rep(0, length = n)
    for(fold_id in 1 : nfold){
        fold_id_vec[fold_id_list[[fold_id]]] <- fold_id
    }

    # related variables for cv results
    loglik_test_mat <- matrix(0, nrow = nfold, ncol = lambda_length)    # store the loglik on the test set
    # each row for one test set
    if(post_selection){
        loglik_post_mat <- loglik_test_mat
    }

    pb <- progressr::progressor(along = 1 : (nfold + 1))    # including the final estimation
    cv_res <- future.apply::future_lapply(1 : nfold, function(cv_id, x_mat, y_vec, h, kn, p,
                                                              p_type, p_param, lambda_seq, mu2,
                                                              a, bj_vec, cj_vec, rj_vec,
                                                              weight_vec, logit_weight_vec, weight_already_combine,
                                                              post_selection, post_a, fold_id_vec){
        loglik_test_mat <- matrix(data = NA, nrow = 2, ncol = length(lambda_seq))
        rownames(loglik_test_mat) <- c("original", "post_selection")
        # get trainging and testing data
        test_id_vec <- which(fold_id_vec == cv_id)
        x_mat_train <- x_mat[-test_id_vec, , drop = FALSE]
        y_vec_train <- y_vec[-test_id_vec]
        weight_vec_train <- weight_vec[-test_id_vec]
        logit_weight_vec_train <- logit_weight_vec[-test_id_vec]
        x_mat_test <- x_mat[test_id_vec, , drop = FALSE]
        y_vec_test <- y_vec[test_id_vec]
        weight_vec_test <- weight_vec[test_id_vec]
        logit_weight_vec_test <- logit_weight_vec[test_id_vec]

        # find solution path on the training set
        print(paste("Find solution path on training set..."))
        train_res <- Logistic_FAR_Path(y_vec = y_vec_train, x_mat = x_mat_train,
                                       h = h, kn = kn, p = p, p_type = p_type, p_param = p_param,
                                       lambda_seq = lambda_seq, mu2 = mu2,
                                       a = a, bj_vec = bj_vec, cj_vec = cj_vec, rj_vec = rj_vec,
                                       weight_vec = weight_vec_train,
                                       logit_weight_vec = logit_weight_vec_train,
                                       weight_already_combine = weight_already_combine,
                                       tol = tol, max_iter = max_iter)

        # test performance on the test set
        print(paste("Compute loglik on the testing set..."))
        for(lam_id in 1 : lambda_length){
            delta_vec <- train_res$delta_path[lam_id, ]
            eta_stack_vec <- train_res$eta_stack_path[lam_id, ]
            # test_pi_vec <- as.vector(x_mat_test %*% c(delta_vec, eta_stack_vec))
            # loglik_test_mat[1, lam_id] <- sum(y_vec_test * test_pi_vec - log(1 + exp(test_pi_vec)))
            test_pi_vec <- as.vector((x_mat_test[, 1 : h, drop = FALSE] %*% delta_vec) + (x_mat_test[, -(1 : h), drop = FALSE] %*% eta_stack_vec) * logit_weight_vec_test)
            loglik_test_mat[1, lam_id] <- sum((y_vec_test * test_pi_vec - log(1 + exp(test_pi_vec))) * weight_vec_test)
        }

        # test on testing set based on post-selection estimation
        if(post_selection){
            # post_res <- train_res
            for(lam_id in 1 : lambda_length){
                post_est <-  Logistic_FAR_Path_Further_Improve(x_mat = x_mat_train, y_vec = y_vec_train, h = h, k_n = kn, p = p,
                                                               delta_vec_init = train_res$delta_path[lam_id, ],
                                                               eta_stack_init = train_res$eta_stack_path[lam_id, ],
                                                               mu1_vec_init = train_res$mu_1_path[lam_id, ],
                                                               # mu1_vec_init = rep(0, k_n),
                                                               mu2 = mu2, a = post_a,
                                                               weight_vec = weight_vec_train,
                                                               logit_weight_vec = logit_weight_vec_train,
                                                               weight_already_combine = weight_already_combine,
                                                               lam = 0.001, tol = 10^{-5}, max_iter = 1000)
                delta_vec <- post_est$delta_vec
                peta_stack_vec <- post_est$eta_stack_vec
                # test_pi_vec <- as.vector(x_mat_test %*% c(delta_vec, eta_stack_vec))
                # loglik_test_mat[2, lam_id] <- sum(y_vec_test * test_pi_vec - log(1 + exp(test_pi_vec)))
                test_pi_vec <- as.vector((x_mat_test[, 1 : h, drop = FALSE] %*% delta_vec) + (x_mat_test[, -(1 : h), drop = FALSE] %*% eta_stack_vec) * logit_weight_vec_test)
                loglik_test_mat[2, lam_id] <- sum((y_vec_test * test_pi_vec - log(1 + exp(test_pi_vec))) * weight_vec_test)
            }
        }

        print(paste(nfold, "-fold CV, FINISHED at ", cv_id, "/", nfold, sep = ""))
        pb(paste(nfold, "-fold CV, folder id = ", cv_id, " finished at pid = ", Sys.getpid(), "!", sep = ""))

        return(loglik_test_mat)

    }, x_mat = x_mat_bak, y_vec = y_vec, h = h, kn = kn, p = p,
    p_type = p_type, p_param = p_param, lambda_seq = lambda_seq, mu2 = mu2,
    a = a, bj_vec = bj_vec, cj_vec = cj_vec, rj_vec = rj_vec,
    weight_vec = weight_vec, logit_weight_vec = logit_weight_vec, weight_already_combine = weight_already_combine,
    post_selection = post_selection, post_a = post_a, fold_id_vec = fold_id_vec)

    ### --- construct the cv result --- ###
    for(cv_id in 1 : nfold){
        cv_res_small <- cv_res[[cv_id]]
        loglik_test_mat[cv_id, ] <- cv_res_small[1, ]
        if(post_selection){
            loglik_post_mat[cv_id, ] <- cv_res_small[2, ]
        }
    }


    # find the lambda with the highest test loglik
    lam_id <- which.max(colSums(loglik_test_mat))
    res <- Logistic_FAR_Path(y_vec = y_vec, x_mat = x_mat_bak,
                             h = h, kn = kn, p = p, p_type = p_type, p_param = p_param,
                             lambda_seq = lambda_seq, mu2 = mu2,
                             a = a, bj_vec = bj_vec, cj_vec = cj_vec, rj_vec = rj_vec,
                             weight_vec = weight_vec, logit_weight_vec = logit_weight_vec,
                             weight_already_combine = weight_already_combine,
                             tol = tol, max_iter = max_iter)

    pb(paste("Computing solution path on the original dataset!"))
    res$cv_id <- lam_id
    res$loglik_test_mat <- loglik_test_mat
    res$fold_id_vec <- fold_id_vec

    if(post_selection){
        # print("post selection")
        lam_post_id <- which.max(colSums(loglik_post_mat))
        # print(paste("lam_post_id = ", lam_post_id, sep = ""))
        # print(paste("delta_vec = ", res$delta_path[lam_post_id, ], sep = ""))
        post_est <- Logistic_FAR_Path_Further_Improve(x_mat = x_mat_bak, y_vec = y_vec, h = h, k_n = kn, p = p,
                                                      delta_vec_init = res$delta_path[lam_post_id, ],
                                                      eta_stack_init = res$eta_stack_path[lam_post_id, ],
                                                      mu1_vec_init = res$mu_1_path[lam_post_id, ],
                                                      # mu1_vec_init = rep(0, k_n),
                                                      mu2 = mu2, a = post_a,
                                                      weight_vec = weight_vec,
                                                      logit_weight_vec = logit_weight_vec,
                                                      weight_already_combine = weight_already_combine,
                                                      lam = 0.001, tol = 10^{-5}, max_iter = 1000)
        res$cv_post_id <- lam_post_id
        res$loglik_post_mat <- loglik_post_mat
        res$post_est <- post_est
    }
    return(res)
}
