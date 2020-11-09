Logistic_FAR_Path2 <- function(y_vec, x_mat, h, k_n, p,
                               p_type, p_param,
                               lambda_seq, lambda_length, min_lambda_ratio = 0.01,
                               mu_2, a = 1, bj_vec = rep(1 / sqrt(k_n), p),
                               h_inv, eta_inv_stack, relax_vec,
                               delta_init, eta_stack_init, mu_1_init,
                               tol, max_iter){
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
    if((h + k_n * p) != ncol(x_mat)){
        stop("supplied h, k_n or p don't match with column number of x_mat!")
    }

    # standardize those grouped covariates in x_mat
    x_mat_bak <- x_mat    # a back up of x_mat
    # transformation matrix, stacked in row
    t_mat_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
    for(i in 1 : p){
        start_idx <- 1 + h + (i - 1) * k_n
        stop_idx <- k_n + h + (i - 1) * k_n
        svd_res <- svd(x_mat[, start_idx : stop_idx, drop = FALSE], nu = 0)
        t_mat <- sqrt(a) * svd_res$v %*% diag(1 / svd_res$d, nrow = k_n)
        t_mat_stack[, (start_idx : stop_idx) - h] <- t_mat
        x_mat[, start_idx : stop_idx] <- x_mat[, start_idx : stop_idx] %*% t_mat
    }

    # covariate matrix for non-functional covariates
    delta_mat <- x_mat[, 1 : h, drop = FALSE]
    if(missing(h_inv)){
        h_mat <- 1 / 4 * t(delta_mat) %*% delta_mat
        h_inv <- solve(h_mat)
    }

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
    if(missing(eta_inv_stack)){
        eta_inv_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
        for(j in 1 : p){
            stack_start <- (j - 1) * k_n + 1
            stack_stop <- j * k_n
            x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
            h_mat_j <- 1 / 4 * t(x_mat_j) %*% x_mat_j
            eta_inv_stack[, stack_start : stack_stop] <- solve(4 * h_mat_j)
        }
    }


    if(missing(relax_vec)){
        relax_vec <- rep(1, p)
        # b_eigen_val_vec <- eigen(t(b_mat) %*% b_mat, only.values = TRUE)
        # b_eigen_max <- max(b_eigen_val_vec$values)
        for(j in 1 : p){
            x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
            h_mat_j <- 1 / 4 * t(x_mat_j) %*% x_mat_j
            eigen_value_vec <- eigen(h_mat_j, only.values = TRUE)
            # origin version of relax vector, where penalty kernel is \theta\eta
            # eigen_min <- min(eigen_value_vec$values)
            # relax_vec[j] <- (1 + 10^(-6)) * (1 + mu_2 * b_eigen_max / eigen_min)
            # new version of relax vector, where penalty kernel is \eta
            eigen_max <- max(eigen_value_vec$values)
            relax_vec[j] <- (1 + 10 ^(-6)) * (mu_2 + eigen_max / a)
        }
        print("Relax vector is: ")
        print(relax_vec)
    }

    if(missing(lambda_seq)){
        print("lambda sequence is missing, using default method to determine it!")

        if(missing(lambda_length) || missing(min_lambda_ratio)){
            stop("Both lambda_length and min_lambda_ratio must be provided for computing the lambda sequence!")
        }else{
            print(paste("lambda_length = ", lambda_length, sep = ""))
            print(paste("min_lambda_ratio = ", min_lambda_ratio, sep = ""))

            # find lambda_max
            # conduct the ordinary logistic regressoin
            logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            pi_fit <- exp(logit_fit$fitted.values) / (1 + exp(logit_fit$fitted.values))
            alpha_vec <- rep(0, p)
            for(i in 1 : p){
                start_ind <- ind_mat[i, 1]
                stop_ind <- ind_mat[i, 2]
                theta_i_mat <- x_mat[, start_ind : stop_ind, drop = FALSE]
                can_vec <- t(theta_i_mat) %*% (y_vec - pi_fit) / a
                # origin version of relax vector, where penalty kernel is \theta\eta
                # alpha_vec[i] <- sqrt(n * t(can_vec) %*% solve(t(theta_i_mat) %*% theta_i_mat) %*% can_vec)

                # new version of relax vector, where penalty kernel is \eta
                alpha_vec[i] <- sqrt(t(can_vec) %*% can_vec) / bj_vec[i]
            }
            rm(theta_i_mat)
            rm(can_vec)
            lam_max = max(alpha_vec)
            rm(alpha_vec)

            lam_min = lam_max * min_lambda_ratio
            # lam_min = lam_max
            lambda_seq <- exp(seq(from = log(lam_max), to = log(lam_min), length.out = lambda_length))

            # default initial values for lambda_max
            print("Using default lambda sequences and initial values for the path searching!")
            # delta_init <- rep(0, h)
            delta_init <- logit_fit$coefficients
            eta_stack_init <- rep(0, p * k_n)
            # mu_1_init <- rep(0, nrow(b_mat))
            mu_1_init <- rep(0, k_n)
        }
    }else{
        lambda_length <- length(lambda_seq)
        lambda_seq <- sort(lambda_seq, decreasing = TRUE)
    }

    delta_path <- matrix(0, nrow = lambda_length, ncol = h)
    eta_stack_path <- matrix(0, nrow = lambda_length, ncol = p * k_n)
    # mu_1_path <- matrix(0, nrow = lambda_length, ncol = nrow(b_mat))
    mu_1_path <- matrix(0, nrow = lambda_length, ncol = k_n)
    iter_num_path <- rep(0, lambda_length)
    converge_path <- rep(0, lambda_length)
    loss_drop_path <- rep(0, lambda_length)

    for(lam_ind in 1 : lambda_length){
        # update lambda
        lambda <- lambda_seq[lam_ind]
        p_param[1] <- lambda

        # conduct the algorithm
        FAR_res <- Logistic_FAR_Solver_Core(y_vec = y_vec, x_mat = x_mat, h = h, kn = k_n, p = p, p_type = p_type, p_param = p_param,
                                            mu2 = mu_2, a = a, bj_vec = bj_vec, tol = tol, max_iter = max_iter, h_inv = h_inv,
                                            relax_vec = relax_vec, delta_init = delta_init, eta_stack_init = eta_stack_init, mu1_init = mu_1_init)
        # save the result
        delta_path[lam_ind, ] <- FAR_res$delta
        eta_stack_path[lam_ind, ] <- FAR_res$eta_stack
        mu_1_path[lam_ind, ] <- FAR_res$mu_1_vec
        iter_num_path[lam_ind] <- FAR_res$iter_num
        converge_path[lam_ind] <- FAR_res$converge
        loss_drop_path[lam_ind] <- FAR_res$loss_drop

        # update initial values for the next run
        delta_init <- FAR_res$delta
        eta_stack_init <- FAR_res$eta_stack
        mu_1_init <- FAR_res$mu_1_vec
        # print some information
        print(paste("Lambda ID = ", lam_ind, ", lambda = ", lambda, " finished!", sep = ""))
    }

    # get the original eta_stack_path
    for(i in 1 : p){
        start_idx <- 1 + (i - 1) * k_n
        stop_idx <- k_n + (i - 1) * k_n
        t_mat <- t_mat_stack[, start_idx : stop_idx, drop = FALSE]
        eta_j_mat <- eta_stack_path[, start_idx : stop_idx, drop = FALSE]
        eta_stack_path[, start_idx : stop_idx] <- t(t_mat %*% t(eta_j_mat))
    }
    # what should we do about the mu_1_path?

    # return the result
    res <- list(delta_path = delta_path,
                eta_stack_path = eta_stack_path,
                mu_1_path = mu_1_path,
                iter_num_path = iter_num_path,
                converge_path = converge_path,
                loss_drop_path = loss_drop_path,
                lambda_seq = lambda_seq)
    return(res)
}


Logistic_FAR_CV_path <- function(y_vec, x_mat, h, k_n, p,
                                 p_type, p_param,
                                 lambda_seq, lambda_length, min_lambda_ratio = 0.01,
                                 mu_2, a = 1, bj_vec = rep(1 / sqrt(k_n), p),
                                 h_inv, eta_inv_stack, relax_vec,
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
    if((h + k_n * p) != ncol(x_mat)){
        stop("supplied h, k_n or p don't match with column number of x_mat!")
    }

    # standardize those grouped covariates in x_mat
    x_mat_bak <- x_mat    # a back up of x_mat
    # transformation matrix, stacked in row
    t_mat_stack <- matrix(0, nrow = kn, ncol = kn * p)
    for(i in 1 : p){
        start_idx <- 1 + h + (i - 1) * kn
        stop_idx <- kn + h + (i - 1) * kn
        svd_res <- svd(x_mat[, start_idx : stop_idx, drop = FALSE], nu = 0)
        t_mat <- sqrt(a) * svd_res$v %*% diag(1 / svd_res$d, nrow = kn)
        t_mat_stack[, (start_idx : stop_idx) - h] <- t_mat
        x_mat[, start_idx : stop_idx] <- x_mat[, start_idx : stop_idx] %*% t_mat
    }

    # covariate matrix for non-functional covariates
    # delta_mat <- x_mat[, 1 : h, drop = FALSE]
    # if(missing(h_inv)){
    #   h_mat <- 1 / 4 * t(delta_mat) %*% delta_mat
    #   h_inv <- solve(h_mat)
    # }

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
    if(missing(eta_inv_stack)){
        eta_inv_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
        for(j in 1 : p){
            stack_start <- (j - 1) * k_n + 1
            stack_stop <- j * k_n
            x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
            h_mat_j <- 1 / 4 * t(x_mat_j) %*% x_mat_j
            eta_inv_stack[, stack_start : stack_stop] <- solve(4 * h_mat_j)
        }
    }

    if(missing(lambda_seq)){
        print("lambda sequence is missing, using default method to determine it!")

        if(missing(lambda_length) || missing(min_lambda_ratio)){
            stop("Both lambda_length and min_lambda_ratio must be provided for computing the lambda sequence!")
        }else{
            print(paste("lambda_length = ", lambda_length, sep = ""))
            print(paste("min_lambda_ratio = ", min_lambda_ratio, sep = ""))

            # find lambda_max
            # conduct the ordinary logistic regressoin
            logit_fit <- glm(y_vec ~ x_mat[, 1 : h, drop = FALSE] - 1, family = binomial)
            pi_fit <- exp(logit_fit$fitted.values) / (1 + exp(logit_fit$fitted.values))
            alpha_vec <- rep(0, p)
            for(i in 1 : p){
                start_ind <- ind_mat[i, 1]
                stop_ind <- ind_mat[i, 2]
                theta_i_mat <- x_mat[, start_ind : stop_ind, drop = FALSE]
                can_vec <- t(theta_i_mat) %*% (y_vec - pi_fit) / a
                # origin version of relax vector, where penalty kernel is \theta\eta
                # alpha_vec[i] <- sqrt(n * t(can_vec) %*% solve(t(theta_i_mat) %*% theta_i_mat) %*% can_vec)

                # new version of relax vector, where penalty kernel is \eta
                alpha_vec[i] <- sqrt(t(can_vec) %*% can_vec) / bj_vec[i]
            }
            rm(theta_i_mat)
            rm(can_vec)
            lam_max = max(alpha_vec)
            rm(alpha_vec)

            lam_min = lam_max * min_lambda_ratio
            # lam_min = lam_max
            lambda_seq <- exp(seq(from = log(lam_max), to = log(lam_min), length.out = lambda_length))

            # default initial values for lambda_max
            print("Using default lambda sequences and initial values for the path searching!")
            # delta_init <- rep(0, h)
            delta_init <- logit_fit$coefficients
            eta_stack_init <- rep(0, p * k_n)
            # mu_1_init <- rep(0, nrow(b_mat))
            mu_1_init <- rep(0, k_n)
        }
    }else{
        lambda_length <- length(lambda_seq)
        lambda_seq <- sort(lambda_seq, decreasing = TRUE)
    }

    # get fold id
    if(!missing(fold_seed)){
        set.seed(fold_seed)
    }
    fold_id_vec <- sample(rep(seq(nfold), length = n))
    # related variables for cv results
    loglik_test_mat <- matrix(0, nrow = nfold, ncol = lambda_length)    # store the loglik on the test set
    # each row for one test set
    if(post_selection){
        loglik_post_mat <- loglik_test_mat
    }
    for(cv_id in 1 : nfold){
        print(paste(nfold, "-fold CV, starting at ", cv_id, "/", nfold, sep = ""))
        test_id_vec <- which(fold_id_vec == cv_id)
        x_mat_train <- x_mat_bak[-test_id_vec, , drop = FALSE]
        y_vec_train <- y_vec[-test_id_vec]
        x_mat_test <- x_mat_bak[test_id_vec, , drop = FALSE]
        y_vec_test <- y_vec[test_id_vec]

        # find solution path on the training set
        print(paste("Find solution path on training set..."))
        train_res <- Logistic_FAR_Path2(y_vec = y_vec_train, x_mat = x_mat_train,
                                        h = h, k_n = k_n, p = p, p_type = p_type, p_param = p_param,
                                        lambda_seq = lambda_seq, mu_2 = mu_2, a = a, bj_vec = bj_vec,
                                        delta_init = delta_init, eta_stack_init = eta_stack_init, mu_1_init = mu_1_init,
                                        tol = tol, max_iter = max_iter)
        # test performance on the test set
        print(paste("Compute loglik on the testing set..."))
        for(lam_id in 1 : lambda_length){
            delta_vec <- train_res$delta_path[lam_id, ]
            eta_stack_vec <- train_res$eta_stack_path[lam_id, ]
            test_pi_vec <- as.vector(x_mat_test %*% c(delta_vec, eta_stack_vec))
            loglik_test_mat[cv_id, lam_id] <- sum(y_vec_test * test_pi_vec - log(1 + exp(test_pi_vec)))
        }

        # test on testing set based on post-selection estimation
        if(post_selection){
            # post_res <- train_res
            for(lam_id in 1 : lambda_length){
                post_est <-  Logistic_FAR_Path_Further_Improve(x_mat = x_mat_train, y_vec = y_vec_train, h = h, k_n = k_n, p = p,
                                                               delta_vec_init = train_res$delta_path[lam_id, ],
                                                               eta_stack_init = train_res$eta_stack_path[lam_id, ],
                                                               # mu1_vec_init = train_res$mu_1_path[lam_id, ],
                                                               mu1_vec_init = rep(0, k_n),
                                                               mu2 = mu2, a = post_a, lam = 0.001, tol = 10^{-5}, max_iter = 1000)
                # post_res$delta_path[lam_id, ] <- post_est$delta_vec
                # post_res$eta_stack_path[lam_id, ] <- post_est$eta_stack_vec
                # post_res$mu1_path[lam_id, ] <- post_est$mu1_vec
                # post_res$iter_num_path[lam_id] <- post_est$iter_num
                # post_res$converge_path[lam_id] <- post_est$converge

                delta_vec <- post_est$delta_vec
                peta_stack_vec <- post_est$eta_stack_vec
                test_pi_vec <- as.vector(x_mat_test %*% c(delta_vec, eta_stack_vec))
                loglik_post_mat[cv_id, lam_id] <- sum(y_vec_test * test_pi_vec - log(1 + exp(test_pi_vec)))
            }

        }
        print(paste(nfold, "-fold CV, FINISHED at ", cv_id, "/", nfold, sep = ""))
    }

    # find the lambda with the highest test loglik
    lam_id <- which.max(colSums(loglik_test_mat))
    res <- Logistic_FAR_Path2(y_vec = y_vec, x_mat = x_mat_bak,
                              h = h, k_n = k_n, p = p, p_type = p_type, p_param = p_param,
                              lambda_seq = lambda_seq, mu_2 = mu_2, a = a, bj_vec = bj_vec,
                              delta_init = delta_init, eta_stack_init = eta_stack_init, mu_1_init = mu_1_init,
                              tol = tol, max_iter = max_iter)

    res$cv_id <- lam_id
    res$loglik_test_mat <- loglik_test_mat
    res$fold_id_vec <- fold_id_vec

    if(post_selection){
        lam_post_id <- which.max(colSums(loglik_post_mat))
        post_est <- Logistic_FAR_Path_Further_Improve(x_mat = x_mat_bak, y_vec = y_vec, h = h, k_n = k_n, p = p,
                                                      delta_vec_init = res$delta_path[lam_post_id, ],
                                                      eta_stack_init = res$eta_stack_path[lam_post_id, ],
                                                      # mu1_vec_init = res$mu_1_path[lam_post_id, ],
                                                      mu1_vec_init = rep(0, k_n),
                                                      mu2 = mu2, a = post_a, lam = 0.001, tol = 10^{-5}, max_iter = 1000)
        res$cv_post_id <- lam_post_id
        res$loglik_post_mat <- loglik_post_mat
        res$post_est <- post_est
    }
    return(res)
}
