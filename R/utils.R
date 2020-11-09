Logit_Pick <- function(y_vec, x_mat, solution_path, real_logit_vec){
  lam_len <- length(solution_path$lambda_seq)
  res <- rep(0, lam_len)
  for(i in 1 : lam_len){
    logit_vec <- x_mat %*% c(solution_path$delta_path[i, ], solution_path$eta_stack_path[i, ])
    res[i] <- mean((logit_vec - real_logit_vec) ^ 2)
  }
  idx <- which.min(res)
  logit_vec <- x_mat %*% c(solution_path$delta_path[idx, ], solution_path$eta_stack_path[idx, ])
  return(list(idx = idx,
              res = res,
              logit_vec = logit_vec))
}

LogLik_Pick <- function(y_vec, x_mat, solution_path, real_logit_vec){
  lam_len <- length(solution_path$lambda_seq)
  res <- rep(0, lam_len)
  for(i in 1 : lam_len){
    logit_vec <- x_mat %*% c(solution_path$delta_path[i, ], solution_path$eta_stack_path[i, ])
    loglik <- sum(y_vec * logit_vec - log(1 + exp(logit_vec)))
    res[i] <- loglik
  }
  idx <- which.max(res)
  logit_vec <- x_mat %*% c(solution_path$delta_path[idx, ], solution_path$eta_stack_path[idx, ])
  return(list(idx = idx,
              res = res,
              logit_vec = logit_vec))
}

#' Pick the optimal \code{lambda} according to the BIC criteria.
#'
#' This function picks the optimal \code{lambda} in a solution path using the
#' BIC criteria.
#'
#' @param y_vec response vector, 0 for control, 1 for case.
#' n = length(y_vec) is the number of observations.
#'
#' @param x_mat covariate matrix, consists of two parts.
#' dim(x_mat) = (n, h + p * kn)
#' First h columns are for demographical covariates(can include an intercept term)
#' Rest columns are for p functional covariates, each being represented by a set of basis functions resulting kn covariates.
#'
#' @param solution_path A solution path from function \code{Logistic_FAR_Path}
#'
#' @param real_logit_vec NOT used in this function
#'
#' @param k_n number of basis functions.(This is also number of covariates in each group)
#'
#' @param a a scalar adjusting the loglik in the first part of BIC
#'
#' @param bic_kn a scalar adjusting the model complexsity part of BIC
#'
#' @param complex_bound Numeric, the upper bound of the model complexsity to be considered.
#' If not supplied, all functional covariates will be considered. In the case of \code{p > n},
#' this may lead to model saturation which makes the BIC cirteria favor a much more
#' complex model because it offers near-perfect fitting results on the training set.
#'
#' @section BIC:
#' In this function, BIC is defined as
#' \deqn{
#'   BIC = 1 / a * loglik + df * log(n) / bic_kn
#'   ,
#' }
#' where \code{df} is the degree of freedom of the model. In this case, it's the number
#' of active covariates in the functional part of \code{x_mat}. Since the algorithm form
#' the problem into a group lasso scenario, here the number of active covariates equals
#' to the number of active functional \eqn{x(t)} times the number of basis functions \code{k_n}.
#' @export
BIC_Pick <- function(y_vec, x_mat, solution_path, real_logit_vec, k_n, a = 1, bic_kn = k_n, complex_bound){
  lam_len <- length(solution_path$lambda_seq)
  n <- length(y_vec)
  p <- solution_path$eta_stack_path[1, ] / k_n
  if(missing(complex_bound)){
    complex_bound <- p
  }

  res <- rep(0, lam_len)
  active_num <- rep(0, lam_len)
  for(i in 1 : lam_len){
    logit_vec <- x_mat %*% c(solution_path$delta_path[i, ], solution_path$eta_stack_path[i, ])
    loglik <- sum(y_vec * logit_vec - log(1 + exp(logit_vec))) / a
    k <- sum(solution_path$eta_stack_path[i, ] != 0) / bic_kn
    bic <- k * log(n) - loglik
    res[i] <- bic

    active_num[i] <- sum(apply(
      matrix(solution_path$eta_stack_path[i, ], nrow = k_n),
      2, function(x){
        return(sum(x ^ 2) > 0)
      }
    ))
  }

  id_vec <- which(active_num <= complex_bound)
  if(length(id_vec) == 0){
    idx <- which.min(active_num)
  }else{
    idx <- which.min(res[id_vec])
    idx <- id_vec[idx]
  }

  logit_vec <- x_mat %*% c(solution_path$delta_path[idx, ], solution_path$eta_stack_path[idx, ])
  return(list(idx = idx,
              res = res,
              logit_vec = logit_vec,
              complex_bound = complex_bound))
}

AUC_Pick <- function(y_vec, x_mat, solution_path, real_logit_vec){
  lam_len <- length(solution_path$lambda_seq)
  res <- rep(0, lam_len)
  for(i in 1 : lam_len){
    logit_vec <- x_mat %*% c(solution_path$delta_path[i, ], solution_path$eta_stack_path[i, ])
    auc_res <- auc(y_vec ~ as.vector(logit_vec), quiet = TRUE)
    res[i] <- auc_res
  }
  idx <- which.max(res)
  logit_vec <- x_mat %*% c(solution_path$delta_path[idx, ], solution_path$eta_stack_path[idx, ])
  return(list(idx = idx,
              res = res,
              logit_vec = logit_vec))
}

#' Pick covariates those enter the solution path first
#'
#' This function picks the solution along the solution path based on a pre-specified number of covariates
#'
#' Note that in practice, the number of selected number of functional covariates
#' might increase more than 1. Therefore it's not uncommon to eventually pick less
#' (or more) than the pre-specified number.
#'
#' @param y_vec response vector, 0 for control, 1 for case.
#' n = length(y_vec) is the number of observations.
#'
#' @param x_mat covariate matrix, consists of two parts.
#' dim(x_mat) = (n, h + p * kn)
#' First h columns are for demographical covariates(can include an intercept term)
#' Rest columns are for p functional covariates, each being represented by a set of basis functions resulting kn covariates.
#'
#' @param solution_path A solution path from function \code{Logistic_FAR_Path}
#'
#' @param real_logit_vec NOT used in this function
#'
#' @param kn number of basis functions.(This is also number of covariates in each group)
#'
#' @param given_number A pre-specified number. This function will pick the \code{given_number}
#' of functional covariates which enter the solution path first.
#'
#' @param upper_bound Logical, default to \code{TRUE}. Whether the \code{given_number}
#' is a strict upper bound. If \code{TRUE}, the picked model will have
#' number of active functional covariates closest to it and never exceeds it.
#'
#' @export
Number_Pick <- function(y_vec, x_mat, solution_path, real_logit_vec, kn, given_number = 5, upper_bound = TRUE){
  # This function picks the solution along the solution path based on a pre-specified number of covariates
  # Note that in practice, the number of selected number of functional covariates might increase more than 1
  #  in the next lambda. So there comes the necessay of the `upper_bound` parameter.
  # Upper_bound = True: the 'given_number' is a strict upper bound, the picked model will have
  #                      number of active functional covariates closest to it and never exceeds it.

  lam_len <- length(solution_path$lambda_seq)
  res_loglik <- rep(0, lam_len)
  res_num <- rep(0, lam_len)
  for(i in 1 : lam_len){
    logit_vec <- x_mat %*% c(solution_path$delta_path[i, ], solution_path$eta_stack_path[i, ])
    loglik <- sum(y_vec * logit_vec - log(1 + exp(logit_vec)))
    res_loglik[i] <- loglik

    eta_mat <- matrix(solution_path$eta_stack_path[i, ], nrow = kn)
    eta_num <- sum(apply(eta_mat, 2, function(x){
      beta_norm <- sum(x ^ 2)
      return(beta_norm != 0)
    }))
    res_num[i] <- eta_num
  }

  if(upper_bound){
    # the given_number is a strong threshholding point
    # the picked number must not exceed it
    id_vec <- which(res_num <= given_number)
    if(length(id_vec) == 0){
      idx <- which.min(res_num)
    }else{
      min_diff <- min(given_number - res_num[id_vec])
      id_vec2 <- which((given_number - res_num[id_vec]) == min_diff)
      idx <- which.max(res_loglik[id_vec[id_vec2]])
      idx <- id_vec[id_vec2[idx]]
    }
  }else{
    # the picked number might exceed it
    id_vec <- which(res_num >= given_number)
    if(length(id_vec) == 0){
      idx <- which.max(res_num)
    }else{
      min_diff <- min(res_num[id_vec] - given_number)
      id_vec2 <- which((res_num[id_vec] - given_number) == min_diff)
      idx <- which.max(res_loglik[id_vec[id_vec2]])
      idx <- id_vec[id_vec2[idx]]
    }
  }

  logit_vec <- x_mat %*% c(solution_path$delta_path[idx, ], solution_path$eta_stack_path[idx, ])
  return(list(
    idx = idx,
    res = res_num,
    res_loglik = res_loglik,
    logit_vec = logit_vec
  ))
}

#' Pick an optimal lambda from a CV solution path
#'
#' This function picks a best estimation from a CV solution path.
#'
#' Although the solver function will always return with a selected lambda
#' This function offers more selecting options.
#'
#' @param y_vec response vector, 0 for control, 1 for case.
#' n = length(y_vec) is the number of observations.
#'
#' @param x_mat covariate matrix, consists of two parts.
#' dim(x_mat) = (n, h + p * kn)
#' First h columns are for demographical covariates(can include an intercept term)
#' Rest columns are for p functional covariates, each being represented by a set of basis functions resulting kn covariates.
#'
#' @param cv_solution_path A solution path and related cross validation information.
#' This is the result from \code{Logistic_FAR_CV_opath}, \code{Logistic_FAR_CV_opath_par}
#' or \code{Logistic_FAR_CV_Path}.
#'
#' @param real_logit_vec Not used in this function
#'
#' @param kn number of basis functions for each functional covariates.
#'
#' @param complex_bound The upper bound for number of active functional covariates to be considered.
#' If missing, the whole path will be considered.
#'
#' @param cv_1se Logical. Whether the 1se strategy be applied.
#'
#' @section 1se Strategy:
#' largest value of lambda such that error is within 1 standard error of the maximum likelihood based on CV
#'
#' @export
CV_Pick <- function(y_vec, x_mat, cv_solution_path, real_logit_vec, kn, complex_bound, cv_1se = FALSE){
  # This function picks a lambda from a solution path resulting from CV path solver
  # Also the solver function will always return with a selected lambda
  # This function offers more selecting options.
  # complex_bound: the upper bound for number of active functional covariates to be considered
  #                original CV solution path will consider all results along the path
  # cv_1se: logical, should the 1se strategy be applied.
  #         1se strategy: largest value of lambda such that error is within 1 standard error of the maximum likelihood based on CV

  lam_len <- length(cv_solution_path$lambda_seq)
  p <- length(cv_solution_path$eta_stack_path[1, ]) / kn    # number of functional covariates
  if(missing(complex_bound)){
    # default value for complex_bound is the number of all functional covariates
    complex_bound <- p
  }

  # find the number of active functional covariates along the solution path
  res_num <- apply(cv_solution_path$eta_stack_path, 1, function(x, kn){
    eta_mat <- matrix(x, nrow = kn)
    active_num <- sum(
      apply(eta_mat, 2, function(x){
        beta_norm <- sum(x ^ 2)
        return(beta_norm > 0)
      }))
    return(active_num)
  }, kn = kn)
  # perform thresh hold according to complex upper bound
  id_vec <- which(res_num <= complex_bound)


  if(length(id_vec) > 0){
    loglik_mat <- cv_solution_path$loglik_test_mat[, id_vec, drop = FALSE]
    if(cv_1se){
      if(length(id_vec) >= 2){
        loglik_sd <- sd(colSums(loglik_mat))
      }else{
        loglik_sd <- 0
      }
      loglik_max <- max(colSums(loglik_mat))
      id_1se <- which(abs(colSums(loglik_mat) - loglik_max) <= loglik_sd)

      # if lambda_seq is in order from big to small
      cv1se_id <- id_vec[min(id_1se)]

      # # if lambda is not in order from big to small
      # lam_vec_1se <- cv_solution_path$lambda_seq[id_vec[id_1se]]
      # lam_id_1se <- which.max(lam_vec_1se)
      # cv1se_id <- id_vec[lam_id_1se]

      cv_id <- which.max(colSums(loglik_mat))
      cv_id <- id_vec[cv_id]

    }else{
      cv_id <- which.max(colSums(loglik_mat))
      cv_id <- id_vec[cv_id]
    }
  }else{
    cv1se_id <- cv_id <- 1
  }

  res <- list(
    idx = cv_id,
    cv_id = cv_id,
    cv1se_id = cv1se_id,
    active_num = res_num,
    complex_bound = complex_bound,
    cv_1se = cv_1se,
    cv_solution_path = cv_solution_path
  )

  if(!is.null(cv_solution_path$cv_post_id)){
    # there are post-selection results during CV
    if(length(id_vec) > 0){
      loglik_mat <- cv_solution_path$loglik_post_mat[, id_vec, drop = FALSE]
      if(cv_1se){
        if(length(id_vec) >= 2){
          loglik_sd <- sd(colSums(loglik_mat))
        }else{
          loglik_sd <- 0
        }
        loglik_max <- max(colSums(loglik_mat))
        id_1se <- which(abs(colSums(loglik_mat) - loglik_max) <= loglik_sd)

        # if lambda_seq is in order from big to small
        cv1se_post_id <- id_vec[min(id_1se)]

        # # if lambda is not in order from big to small
        # lam_vec_1se <- cv_solution_path$lambda_seq[id_vec[id_1se]]
        # lam_id_1se <- which.max(lam_vec_1se)
        # cv1se_post_id <- id_vec[lam_id_1se]

        cv_post_id <- which.max(colSums(loglik_mat))
        cv_post_id <- id_vec[cv_post_id]
      }else{
        cv_post_id <- which.max(colSums(loglik_mat))
        cv_post_id <- id_vec[cv_post_id]
      }
    }else{
      cv1se_post_id <- cv_post_id <- 1
    }
    res$cv1se_post_id = cv1se_post_id
    res$cv_post_id = cv_post_id
  }

  return(res)
}


Confusion_Mat <- function(eta_stack_vec, k_n, pos_id_vec, neg_id_vec){
  eta_mat <- matrix(eta_stack_vec, nrow = k_n)
  p <- ncol(eta_mat)
  pos_check <- apply(eta_mat, 2, FUN = function(x){
    res <- (sum(x != 0) > 0)
    return(res)
  })
  pos_id <- which(pos_check == TRUE)
  neg_id <- which(pos_check == FALSE)
  TP <- sum(is.element(pos_id, pos_id_vec))
  TN <- sum(is.element(neg_id, neg_id_vec))
  FN <- length(pos_id_vec) - TP
  FP <- length(neg_id_vec) - TN
  return(list(TP = TP,
              TN = TN,
              FP = FP,
              FN = FN))
}

#' @export
Summary_Simulation_Res <- function(delta_mat, eta_mat, logit_mse_vec, delta0, eta_vec0, k_n){
  # sim_num <- nrow(delta_mat)
  res <- matrix(0, nrow = 2, ncol = 6)
  colnames(res) <- c("MSE_Logit", "MSE_Delta", "MSE_Eta", "FP", "FN", "FDR")
  rownames(res) <- c("mean", "sd")

  res["mean", "MSE_Logit"] <- mean(logit_mse_vec)
  res["sd", "MSE_Logit"] <- sd(logit_mse_vec)

  delta_mse <- apply(delta_mat, 1, FUN = function(x, vec0){
    return(mean((x - vec0) ^ 2))
  }, vec0 = delta0)
  eta_mse <- apply(eta_mat, 1, FUN = function(x, vec0){
    return(mean((x - vec0) ^ 2))
  }, vec0 = eta_vec0)
  res["mean", "MSE_Delta"] <- mean(delta_mse)
  res["sd", "MSE_Delta"] <- sd(delta_mse)
  res["mean", "MSE_Eta"] <- mean(eta_mse)
  res["sd", "MSE_Eta"] <- sd(eta_mse)

  eta_mat0 <- matrix(eta_vec0, nrow = k_n)
  p <- ncol(eta_mat0)
  pos_check <- apply(eta_mat0, 2, FUN = function(x){
    res <- sum(x != 0) > 0
    return(res)
  })
  pos_id_vec <- which(pos_check == TRUE)
  neg_id_vec <- which(pos_check == FALSE)
  fp_vec <- apply(eta_mat, 1, FUN = function(x, k_n, pos_id_vec, neg_id_vec){
    res <- Confusion_Mat(x, k_n, pos_id_vec, neg_id_vec)
    return(res$FP)
  }, k_n = k_n, pos_id_vec = pos_id_vec, neg_id_vec = neg_id_vec)
  fn_vec <- apply(eta_mat, 1, FUN = function(x, k_n, pos_id_vec, neg_id_vec){
    res <- Confusion_Mat(x, k_n, pos_id_vec, neg_id_vec)
    return(res$FN)
  }, k_n = k_n, pos_id_vec = pos_id_vec, neg_id_vec = neg_id_vec)
  fdr_vec <- apply(eta_mat, 1, FUN = function(x, k_n, pos_id_vec, neg_id_vec){
    res <- Confusion_Mat(x, k_n, pos_id_vec, neg_id_vec)
    if((res$TP + res$FP) == 0){
      fdr_res <- 0
    }else{
      fdr_res <- res$FP / (res$FP + res$TP)
    }
    return(fdr_res)
  }, k_n = k_n, pos_id_vec = pos_id_vec, neg_id_vec = neg_id_vec)
  res["mean", "FP"] <- mean(fp_vec)
  res["sd", "FP"] <- sd(fp_vec)
  res["mean", "FN"] <- mean(fn_vec)
  res["sd", "FN"] <- sd(fn_vec)
  res["mean", "FDR"] <- mean(fdr_vec)
  res["sd", "FDR"] <- sd(fdr_vec)

  return(res)
}


Compute_Loss <- function(x_mat, y_vec, delta_vec, eta_stack_vec, mu1_vec, mu_2, h, kn, p, p_type, p_param, a = 1, bj_vec = rep(1 / sqrt(kn), p), oracle_loss = FALSE, print_res = TRUE){
    # loss part 0: loglik
    logit_vec <- x_mat %*% c(delta_vec, eta_stack_vec)
    loglik <- sum(y_vec * logit_vec - log(1 + exp(logit_vec)))
    loglik <- loglik / a

    # loss part 2: ADMM
    eta_mat <- matrix(eta_stack_vec, nrow = kn)
    loss_p2 <- t(mu1_vec) %*% rowSums(eta_mat) + mu_2 / 2 * t(rowSums(eta_mat)) %*% rowSums(eta_mat)
    loss <- -loglik + loss_p2

    if(oracle_loss){
        # there is no group lasso penalty in oracle loss function
        if(print_res){
            print(paste("1 / a * loglik = ", loglik, ", loss_p2 = ", loss_p2, ", loss = ", loss, sep = ""))
        }
        return(loss)
    }else{    # loss part 1: penalty
        # determine penalty function
        if(p_type == "L"){
            pfun <- penalty_lasso
        }else{
            if(p_type == "S"){
                pfun <- penalty_scad
            }else{
                if(p_type == "M"){
                    pfun <- penalty_mcp
                }else{
                    stop("p_type must be among 'L', 'S', or 'M'!")
                }
            }
        }

        # ind_mat stores the starting and stopping index for each functional covariates
        #   in x_mat. Each row for one functional covariates.
        # ind_mat also provides starting and stopping indces in eta_stack_vec, just minus h
        #   since there's no delta part in eta.
        # ind_mat <- matrix(0, nrow = p, ncol = 2)
        # colnames(ind_mat) <- c("start_ind", "stop_ind")
        # rownames(ind_mat) <- paste("v", 1 : p, sep = "")
        # ind_mat[, 1] <- (0 : (p - 1)) * kn + 1 + h
        # ind_mat[, 2] <- (1 : p) * kn + h

        n <- length(y_vec)

        # origin version, where penalty kernel is \theta\eta
        # tmp <- matrix(0, nrow = n, ncol = p)
        # for(j in 1 : p){
        #   x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
        #   tmp[, j] <- x_mat_j %*% eta_mat[, j]
        # }
        # loss_p1 <- apply(tmp, 2, function(x){
        #   return(sqrt(t(x) %*% x))
        # })
        # loss_p1 <- lambda / sqrt(n) * sum(loss_p1)

        # new version, where penalty kernel is \eta
        tmp <- eta_mat
        loss_p1 <- apply(tmp, 2, function(x){
            # l2 norm of every eta vector
            return(sqrt(t(x) %*% x))
        })

        loss_p1 <- sum(pfun(x = loss_p1 * bj_vec, params = p_param, derivative = FALSE))

        loss <- loss + loss_p1
        if(print_res){
            print(paste("1 / a * loglik = ", loglik, ", loss_p1 = ", loss_p1, ", loss_p2 = ", loss_p2, ", loss = ", loss, sep = ""))
        }
        return(loss)
    }
}

#' Post-selection estimation
#'
#' This function performs post-selection estimation on a given solution.
#'
#' @param x_mat covariate matrix, consists of two parts.
#' dim(x_mat) = (n, h + p * kn)
#' First h columns are for demographical covariates(can include an intercept term)
#' Rest columns are for p functional covariates, each being represented by a set of basis functions resulting kn covariates.
#'
#' @param y_vec response vector, 0 for control, 1 for case.
#' n = length(y_vec) is the number of observations.
#'
#' @param h,k_n,p dimension information for the dataset(\code{x_mat}).
#'
#' @param delta_vec_init,eta_stack_init,mu1_vec_init Initial values for the algorithm.
#' This function uses these initial values to find out the active functional covariates.
#' And the post-selection estimation begins with these initial values.
#'
#' @param mu2 quadratic term in the ADMM algorithm
#'
#' @param a parameters for the algorithm. The 1st term in the loss function is
#' \code{1 / a * loglik}. See Algorithm_Details.pdf
#' for more information.
#'
#' @param lam A scalar for the regularize in ridge penalty form in case of model saturation.
#'
#' @param tol,max_iter convergence tolerance and max number of iteration of the algorithm.
#'
#' @export
Logistic_FAR_Path_Further_Improve <- function(x_mat, y_vec, h, k_n, p, delta_vec_init, eta_stack_init, mu1_vec_init, mu2, a = 1, lam = 0.1, tol = 10^(-5), max_iter = 1000){
    # Post selection estimation to further improve the estimation from a solution path
    # Args: x_mat
    #       y_vec
    #       h
    #       k_n
    #       p
    #       delta_vec_init
    #       eta_stack_init
    #       mu1_vec_init
    #       mu2
    #       a
    #       lam

    ######------------ prepare the data ------------
    y_vec <- as.vector(y_vec)
    x_mat <- as.matrix(x_mat)
    n <- length(y_vec)    # number of observations
    if(n != nrow(x_mat)){
        stop("x_mat and y_vec don't have the same number of observations")
    }
    if((h + k_n * p) != ncol(x_mat)){
        stop("supplied h, k_n or p don't match with column number of x_mat!")
    }

    # covariate matrix for non-functional covariates
    delta_mat <- x_mat[, 1 : h, drop = FALSE]
    h_mat <- 1 / 4 * t(delta_mat) %*% delta_mat
    delta_inv <- solve(h_mat / a + lam * diag(nrow = h))    # the inverse matrix for updating delta_vec


    # covariate matrices for functional covariates
    # ind_mat stores the starting and stopping index for each functional covariates
    #   in x_mat. Each row for one functional covariates.
    # ind_mat also provides starting and stopping indces in eta_stack_vec, just minus h
    #   since there's no delta part in eta.
    ind_mat <- matrix(0, nrow = p, ncol = 2)
    colnames(ind_mat) <- c("start_ind", "stop_ind")
    rownames(ind_mat) <- paste("v", 1 : p, sep = "")
    ind_mat[, 1] <- (0 : (p - 1)) * k_n + 1 + h
    ind_mat[, 2] <- (1 : p) * k_n + h

    # find active functional covariates based on eta_stack_init
    eta_mat <- matrix(eta_stack_init, nrow = k_n)
    col_norm <- apply(eta_mat, 2, function(x) sum(x ^ 2))
    active_idx <- which(col_norm != 0)

    ######------------ main algorithm ------------
    diff <- tol + 1
    iter_num <- 1
    delta_vec_old <- delta_vec_init
    eta_stack_vec <- eta_stack_init
    mu1_vec <- mu1_vec_init

    # depends on whether there are active functional covariates.
    if(length(active_idx) == 0){
        # no active functional covariates
        # only update the demographical covariates delta
        while(iter_num <= max_iter && diff > tol){
            logit_vec <- delta_mat %*% delta_vec_old
            pi_vec <- exp(logit_vec) / (1 + exp(logit_vec))
            delta_vec <- 1 / a * delta_inv %*% (h_mat %*% delta_vec_old + t(delta_mat) %*% (y_vec - pi_vec))
            diff <- mean((delta_vec - delta_vec_old) ^ 2)
            iter_num <- iter_num + 1
            delta_vec_old <- delta_vec
        }
    }else{
        # we found some active functional covariates
        # construct the corresponding theta_mat, eta_stack_vec and mu1_vec
        x_active_mat <- matrix(0, nrow = n, ncol = length(active_idx) * k_n)
        eta_active_stack_vec <- rep(0, length(active_idx) * k_n)

        for(i in 1 : length(active_idx)){
            idx <- active_idx[i]
            # start and stop index in the original x_mat
            start_ind <- ind_mat[idx, 1]
            stop_ind <- ind_mat[idx, 2]

            # start and stop index in the resulting vector/matrix
            res_start_ind <- (i - 1) * k_n + 1
            res_stop_ind <- i * k_n

            # copy the data
            x_active_mat[, res_start_ind : res_stop_ind] <- x_mat[, start_ind : stop_ind]
            eta_active_stack_vec[res_start_ind : res_stop_ind] <- eta_stack_init[(start_ind : stop_ind) - h]
        }

        c_mat <- matrix(rep(diag(nrow = k_n), length(active_idx)), nrow = k_n)
        h_mat_eta <- 1 / 4 * t(x_active_mat) %*% x_active_mat
        eta_inv <- solve(1 / a * h_mat_eta + lam * diag(nrow = k_n * length(active_idx)) + mu2 * t(c_mat) %*% c_mat)

        delta_vec_old <- delta_vec_init
        eta_active_stack_vec_old <- eta_active_stack_vec
        mu1_vec_old <- mu1_vec
        while(iter_num <= max_iter && diff > tol){
            # update delta
            logit_vec <- cbind(delta_mat, x_active_mat) %*% c(delta_vec_old, eta_active_stack_vec_old)
            pi_vec <- exp(logit_vec) / (1 + exp(logit_vec))
            delta_vec <- 1 / a * delta_inv %*% (h_mat %*% delta_vec_old + t(delta_mat) %*% (y_vec - pi_vec))

            # update eta
            logit_vec <- cbind(delta_mat, x_active_mat) %*% c(delta_vec, eta_active_stack_vec_old)
            pi_vec <- exp(logit_vec) / (1 + exp(logit_vec))
            eta_active_stack_vec <- eta_inv %*% (1 / a * h_mat_eta %*% eta_active_stack_vec_old + 1 / a * t(x_active_mat) %*% (y_vec - pi_vec) - t(c_mat) %*% mu1_vec_old)

            # update mu1
            eta_mat <- matrix(eta_active_stack_vec, nrow = k_n)
            mu1_vec <- mu1_vec_old + mu2 * rowSums(eta_mat)

            diff1 <- mean((delta_vec - delta_vec_old) ^ 2)
            diff2 <-  mean((eta_active_stack_vec - eta_active_stack_vec_old) ^ 2)
            diff <- max(diff1, diff2)
            iter_num <- iter_num + 1

            delta_vec_old <- delta_vec
            eta_active_stack_vec_old <- eta_active_stack_vec
            mu1_vec_old <- mu1_vec
        }

        # save the result back to original form
        for(i in 1 : length(active_idx)){
            idx <- active_idx[i]
            # start and stop index in the original x_mat
            start_ind <- ind_mat[idx, 1]
            stop_ind <- ind_mat[idx, 2]
            # start and stop index in the resulting active vector/matrix
            res_start_ind <- (i - 1) * k_n + 1
            res_stop_ind <- i * k_n
            # copy the result back to original form
            # print(paste("start_ind = ", start_ind, ", stop_ind = ", stop_ind, sep = ""))
            # print(paste("res_start_ind = ", res_start_ind, ", res_stop_ind = ", res_stop_ind, sep = ""))
            eta_stack_vec[(start_ind : stop_ind) - h] <- eta_active_stack_vec[res_start_ind : res_stop_ind]
        }
    }

    res <- list(delta_vec = delta_vec,
                eta_stack_vec = eta_stack_vec,
                mu1_vec = mu1_vec,
                a = a,
                regular = lam,
                iter_num = iter_num,
                converge = (diff <= tol))
    return(res)
}


NBZI_Confusion_Mat <- function(adjust_p_vec, alpha_level, pos_id_vec, neg_id_vec){
  pos_check <- adjust_p_vec < alpha_level
  pos_id <- which(pos_check == TRUE)
  neg_id <- which(pos_check == FALSE)
  TP <- sum(is.element(pos_id, pos_id_vec))
  TN <- sum(is.element(neg_id, neg_id_vec))
  FN <- length(pos_id_vec) - TP
  FP <- length(neg_id_vec) - TN
  return(list(TP = TP,
              TN = TN,
              FP = FP,
              FN = FN))
}

NBZI_Summary_Simulation <- function(adjust_p_mat, alpha_level, eta_vec0, k_n){
  res <- matrix(0, nrow = 2, ncol = 3)
  colnames(res) <- c("FP", "FN", "FDR")
  rownames(res) <- c("mean", "sd")

  eta_mat0 <- matrix(eta_vec0, nrow = k_n)
  p <- ncol(eta_mat0)
  pos_check <- apply(eta_mat0, 2, FUN = function(x){
    res <- sum(x != 0) > 0
    return(res)
  })
  pos_id_vec <- which(pos_check == TRUE)
  neg_id_vec <- which(pos_check == FALSE)

  fp_vec <- apply(adjust_p_mat, 1,
                  function(x, alpha_level, pos_id_vec, neg_id_vec){
                    res <- NBZI_Confusion_Mat(adjust_p_vec = x,
                                              alpha_level = alpha_level,
                                              pos_id_vec = pos_id_vec,
                                              neg_id_vec = neg_id_vec)
                    return(res$FP)
                  },
                  alpha_level = alpha_level, pos_id_vec = pos_id_vec, neg_id_vec = neg_id_vec)

  fn_vec <- apply(adjust_p_mat, 1,
                  function(x, alpha_level, pos_id_vec, neg_id_vec){
                    res <- NBZI_Confusion_Mat(adjust_p_vec = x,
                                              alpha_level = alpha_level,
                                              pos_id_vec = pos_id_vec,
                                              neg_id_vec = neg_id_vec)
                    return(res$FN)
                  },
                  alpha_level = alpha_level, pos_id_vec = pos_id_vec, neg_id_vec = neg_id_vec)

  fdr_vec <- apply(adjust_p_mat, 1,
                   function(x, alpha_level, pos_id_vec, neg_id_vec){
                     res <- NBZI_Confusion_Mat(adjust_p_vec = x,
                                               alpha_level = alpha_level,
                                               pos_id_vec = pos_id_vec,
                                               neg_id_vec = neg_id_vec)
                     if((res$FP + res$TP) == 0){
                       fdr_res <- 0
                     }else{
                       fdr_res <- res$FP / (res$FP + res$TP)
                     }
                     return(fdr_res)
                   },
                   alpha_level = alpha_level, pos_id_vec = pos_id_vec, neg_id_vec = neg_id_vec)

  res["mean", "FP"] <- mean(fp_vec)
  res["sd", "FP"] <- sd(fp_vec)
  res["mean", "FN"] <- mean(fn_vec)
  res["sd", "FN"] <- sd(fn_vec)
  res["mean", "FDR"] <- mean(fdr_vec)
  res["sd", "FDR"] <- sd(fdr_vec)
  return(res)
}

Gen_Microbiome_Data <- function(n, n_control_proportioin, n_control, t_num, p = 1, p_active = 1,
                                control_mean, sigma, rho, corr_str, func_form, beta, IP = NULL,
                                missing_pct, missing_per_subject, miss_val = 0,
                                zero_trunc = TRUE, asynch_time = FALSE,
                                cl = NULL){
  # generate the simulated dataset using the `mvrnorm_sim` functions in the `microbiomeDASim` package.
  # Args: n: number of subjects
  #       n_control_proportion: proportion of the control group in subjects, n_control = round(n_control_proportion * n)
  #       n_control: size of control group
  #       t_num: number of time points. NOTE: the time interval is always [0, 1]
  #       control_mean, sigma, rho, corr_str, func_form, beta, IP, missing_pct, missing_per_subject, miss_val, zero_trunc, asynch_time: arguments for `mvrnorm_sim`
  #           NOTE: currently, `asynch_time` is always set to `FALSE`
  #                            `zero_trunc` is always set to `TRUE`
  #       p: number of covariates(features)
  #       p_active: number of activte covariates(abundently different features)
  #       cl: cluster from package `parallel`, if supplied, the function will generate data in parallel

  if(missing(n_control)){
    if(missing(n_control_proportioin)){
      stop("At least one of n_control_proportion and n_control should be provided!")
    }else{
      n_control <- round(n * n_control_proportioin)
    }
  }
  if(p < p_active){
    stop("Number of features(p) LESS than active number of features(p_active)!")
  }else{
    p_inactive <- p - p_active
  }
  print(paste("Control group size = ", n_control, ". Treatment group size = ", n - n_control, sep = ""))
  print(paste("Number of active features = ", p_active, ". Number of in-active features = ", p_inactive, sep = ""))
  if(p == 1){    # There are only 1 feature to generate
    if(p_active == 1){    # there are only 1 active feature to generate
      print("Generating the only 1 active feature ...")
      res <- microbiomeDASim::mvrnorm_sim(n_control = n_control, n_treat = n - n_control,
                                          control_mean = control_mean, sigma = sigma,
                                          num_timepoints = t_num, t_interval = c(0, 1),
                                          rho = rho, corr_str = corr_str, func_form = func_form,
                                          beta = beta, IP = IP,
                                          missing_pct = missing_pct,
                                          missing_per_subject = missing_per_subject,
                                          miss_val = miss_val,
                                          dis_plot = FALSE, plot_trend = FALSE,
                                          zero_trunc = zero_trunc, asynch_time = asynch_time)
    }else{    # there are only 1 in-active feature to generate
      print("Generating the only 1 inactive feature ...")
      res <- microbiomeDASim::mvrnorm_sim(n_control = n_control, n_treat = n - n_control,
                                          control_mean = control_mean, sigma = sigma,
                                          num_timepoints = t_num, t_interval = c(0, 1),
                                          rho = rho, corr_str = corr_str, func_form = "linear",
                                          beta = c(0, 0), IP = IP,
                                          missing_pct = missing_pct,
                                          missing_per_subject = missing_per_subject,
                                          miss_val = miss_val,
                                          dis_plot = FALSE, plot_trend = FALSE,
                                          zero_trunc = zero_trunc, asynch_time = asynch_time)
    }
    res_real <- t(matrix(res$Y, nrow = t_num))
    res_obs <- t(matrix(res$Y_obs, nrow = t_num))
    res_id <- 1 : n
    res_group <- c(rep("control", n_control), rep("treatment", n - n_control))
    rownames(res_real) <- paste(res_id, res_group, sep = "_")
    rownames(res_obs) <- paste(res_id, res_group, sep = "_")
    res_time <- res$df$time[1 : t_num]

    res <- list(real_mat = res_real,
                obs_mat = res_obs,
                ID = res_id,
                group = res_group,
                time_points = res_time)

  }else{    # There are multiple features to generate
    if(p_active > 0){
      print(paste("Generating ", p_active, " active feature(s) ...", sep = ""))
      if(inherits(cl, "cluster")){
        parallel::clusterExport(cl = cl,
                                varlist = c("n_control", "control_mean", "sigma", "t_num", "rho", "corr_str", "func_form", "beta", "IP", "missing_pct", "missing_per_subject", "miss_val", "zero_trunc", "asynch_time"),
                                envir = environment())
      }
      res_active <- pbapply::pblapply(X = seq_len(p_active),
                                      FUN = function(x){
                                        res_feature <- microbiomeDASim::mvrnorm_sim(n_control = n_control, n_treat = n - n_control,
                                                                                    control_mean = control_mean, sigma = sigma,
                                                                                    num_timepoints = t_num, t_interval = c(0, 1),
                                                                                    rho = rho, corr_str = corr_str, func_form = func_form,
                                                                                    beta = beta, IP = IP,
                                                                                    missing_pct = missing_pct,
                                                                                    missing_per_subject = missing_per_subject,
                                                                                    miss_val = miss_val,
                                                                                    dis_plot = FALSE, plot_trend = FALSE,
                                                                                    zero_trunc = zero_trunc, asynch_time = asynch_time)
                                        return(res_feature)
                                      },
                                      cl = cl)

      # unlist the result
      res_active_real <- matrix(0, nrow = n, ncol = t_num * p_active)
      res_active_obs <- matrix(0, nrow = n, ncol = t_num * p_active)

      for(i in 1 : p_active){
        res_active_real[, 1 : t_num + (i - 1) * t_num] <- t(matrix(res_active[[i]]$Y, nrow = t_num))
        res_active_obs[, 1 : t_num + (i - 1) * t_num] <- t(matrix(res_active[[i]]$Y_obs, nrow = t_num))
      }
      res_id <- 1 : n
      res_group <- c(rep("control", n_control), rep("treatment", n - n_control))
      rownames(res_active_real) <- paste(res_id, res_group, sep = "_")
      rownames(res_active_obs) <- paste(res_id, res_group, sep = "_")
      res_time <- res_active[[1]]$df$time[1 : t_num]
    }else{
      print("There are no active features to be generated!")
      res_active <- null
    }

    if(p_inactive > 0){
      print(paste("Generating ", p_inactive, " inactive feature(s) ...", sep = ""))
      if(inherits(cl, "cluster")){
        parallel::clusterExport(cl = cl,
                                varlist = c("n_control", "control_mean", "sigma", "t_num", "rho", "corr_str", "func_form", "beta", "IP", "missing_pct", "missing_per_subject", "miss_val", "zero_trunc", "asynch_time"),
                                envir = environment())
      }
      res_inactive <- pbapply::pblapply(X = seq_len(p_inactive),
                                        FUN = function(x){
                                          res_feature <- microbiomeDASim::mvrnorm_sim(n_control = n_control, n_treat = n - n_control,
                                                                                      control_mean = control_mean, sigma = sigma,
                                                                                      num_timepoints = t_num, t_interval = c(0, 1),
                                                                                      rho = rho, corr_str = corr_str, func_form = "linear",
                                                                                      beta = c(0, 0), IP = IP,
                                                                                      missing_pct = missing_pct,
                                                                                      missing_per_subject = missing_per_subject,
                                                                                      miss_val = miss_val,
                                                                                      dis_plot = FALSE, plot_trend = FALSE,
                                                                                      zero_trunc = zero_trunc, asynch_time = asynch_time)
                                          return(res_feature)
                                        },
                                        cl = cl)

      # unlist the result
      res_inactive_real <- matrix(0, nrow = n, ncol = t_num * p_inactive)
      res_inactive_obs <- matrix(0, nrow = n, ncol = t_num * p_inactive)

      for(i in 1 : p_inactive){
        res_inactive_real[, 1 : t_num + (i - 1) * t_num] <- t(matrix(res_inactive[[i]]$Y, nrow = t_num))
        res_inactive_obs[, 1 : t_num + (i - 1) * t_num] <- t(matrix(res_inactive[[i]]$Y_obs, nrow = t_num))
      }
      res_id <- 1 : n
      res_group <- c(rep("control", n_control), rep("treatment", n - n_control))
      rownames(res_inactive_real) <- paste(res_id, res_group, sep = "_")
      rownames(res_inactive_obs) <- paste(res_id, res_group, sep = "_")
      res_time <- res_inactive[[1]]$df$time[1 : t_num]
    }else{
      print("There are no inactive features to be generated!")
      res_inactive <- null
    }

    # combine active and inactive results
    if(is.null(res_active)){    # there are no active part
      # then there must be an inactive part
      res_real <- res_inactive_real
      res_obs <- res_inactive_obs

    }else{    # there are active part
      if(is.null(res_inactive)){     # there is no inactive part
        res_real <- res_active_real
        res_obs <- res_active_obs

      }else{
        res_real <- cbind(res_active_real, res_inactive_real)
        res_obs <- cbind(res_active_obs, res_inactive_obs)
      }
    }
    res <- list(real_mat = res_real,
                obs_mat = res_obs,
                ID = res_id,
                group = res_group,
                time_points = res_time)
  }

  if(zero_trunc){
    # zero truncation
    # the zero truncation in microbiomeDASim::mvrnorm_sim might be disabled
    id <- which(res$obs_mat < 0)
    if(length(id) > 0){
      res$obs_mat[id] <- 0
    }

    id <- which(res$real_mat < 0)
    if(length(id) > 0){
      res$real_mat[id] <- 0
    }
  }
  return(res)
}
