#' @export
Logistic_FAR_Oracle <- function(y_vec, x_mat, h, k_n, p, mu_2,
                                h_inv, eta_inv_stack,
                                delta_init, eta_stack_init, mu_1_init,
                                tol, max_iter){
    # This function computes the Logistic FAR Oracle estimator.
    # Args: y_vec: 0-1 vector of response, numerical, not factor
    #              n <- length(y_vec) is the number of observations.
    #       x_mat: covariate matrix. Each row for one observation.
    #              First h columns for the non-functional covariates,
    #              while the rest columns are the result of basis function approximation.
    #              These stack together forming the x_mat.
    #              Note: the intercept term is also considered as a non-functional covariate,
    #                    so add an all-one column in x_mat for estimating the intercept effect(the alpha in the notes.)
    #              nrow(x_mat) is the number of observations
    #              ncol(x_mat) = h + k_n * p is the number of covariates
    #              In the original dataset: There are h demographical covariates.
    #                                       And p functional covariates, each represented using a k_n dimensional basis functions.
    #       h: number of demographical covariates (plus a possible intercept term)
    #       NOTE: in my notes, there are alpha(intercept term) and delta(demographical covariates) in the dataset
    #               and length(delta) = h.
    #             But in the algorithm, there's no meanning in discrimate intercept term from demographical covariates.
    #               They share the same updating step.
    #             So if you want the intercept term, then put a 1 column in x_mat; if you don't then don't put it there.
    #             And h in this function represents the number of non-functional covariates in x_mat.
    #       k_n: number of the basis functions
    #       p: number of the functional covariates
    #       h_inv: inverse matrix of non-functional part's H matrix, its size is h * h
    #       eta_inv_stack: inverse matrix stacked in one direction
    #                      currently using row major stack, hence nrow(eta_inv_stack) = k_n, ncol(eta_inv_stack) = k_n * p
    #       delta_init: initial value for delta vector, length(delta_init) = h, the total number of non-functional covariates
    #       eta_stack_init: initial values for eta vecotrs stacked in one vector
    #                       length(eta_stack_init) = p * k_n
    #                       eta_stack_init = c(eta_1, eta_2, ..., eta_p)
    #       mu_1_init: initial value vector for multiplier vector mu_1 in ADMM algorithm
    #       tol: convergence tolerance
    #       max_iter: max number of iteration
    
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
        # for updating eta_j seperately
        eta_inv_stack <- matrix(0, nrow = k_n, ncol = k_n * p)
        for(j in 1 : p){
            stack_start <- (j - 1) * k_n + 1
            stack_stop <- j * k_n
            x_mat_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = FALSE]
            h_mat_j <- 1 / 4 * t(x_mat_j) %*% x_mat_j
            eta_inv_stack[, stack_start : stack_stop] <- solve(h_mat_j + mu_2 * diag(k_n))
            print(eigen(h_mat_j + mu_2 * diag(k_n))$values)
        }
    }
    # for updating eta_j together
    c_mat = matrix(rep(diag(k_n), p), nrow = k_n)
    print(c_mat)
    x_eta_mat <- x_mat[, 1 : (k_n * p) + h, drop = FALSE]
    h_eta <- 1 / 4 * t(x_eta_mat) %*% x_eta_mat
    h_eta_inv <- solve(h_eta + mu_2 * t(c_mat) %*% c_mat + 0.05 * diag(k_n * p))
    
    ###### ------------ main algorithm ------------
    diff <- 1
    current_iter <- 0
    converge <- FALSE
    loss_drop <- TRUE
    
    delta <- delta_init
    eta_stack <- eta_stack_init
    mu_1_vec <- mu_1_init
    eta_mat <- matrix(eta_stack, nrow = k_n)
    
    logit_vec <- x_mat %*% c(delta, eta_stack)
    loglik <- sum(y_vec * logit_vec - log(1 + exp(logit_vec)))
    loss_p2 <- t(mu_1_vec) %*% rowSums(eta_mat) + mu_2 / 2 * t(rowSums(eta_mat)) %*% rowSums(eta_mat)
    loss <- -loglik + loss_p2
    print(paste("Initial loss  = ", loss, sep = ""))
    Compute_Loss(x_mat, y_vec, delta, eta_stack, mu_1_vec, mu_2, h, k_n, p, 0, oracle_loss = TRUE, print_res = T)
    
    
    
    while((!converge) && (current_iter < max_iter) && (loss_drop)){
        # store results from previous iteration
        delta_old <- delta
        eta_stack_old <- eta_stack
        mu_1_vec_old <- mu_1_vec
        loss_old <- loss
        
        # update
        # step 1. get the current pi_vec
        logit_vec <- x_mat %*% c(delta_old, eta_stack_old)
        pi_vec <- exp(logit_vec) / (1 + exp(logit_vec))
        
        # step 2. update the demographical covariates
        delta <- delta_old - h_inv %*% t(delta_mat) %*% (pi_vec - y_vec)
        # print("After updating delta:")
        # Compute_Loss(x_mat, y_vec, delta, eta_stack, mu_1_vec, mu_2, h, k_n, p, 0, oracle_loss = TRUE, print_res = T)
        # update logit_vec and pi_vec
        logit_vec <- x_mat %*% c(delta, eta_stack_old)
        pi_vec <- exp(logit_vec) / (1 + exp(logit_vec))
        
        # step 3. update the functional covariates
        eta_mat_old <- matrix(eta_stack_old, nrow = k_n)    # eta_stack_old in matrix form, k_n * p
        # each COLUMN for one covariates
        
        # update eta_j blockwisely
        # for(j in 1 : p){
        #   # prepare some necessay variables
        #   stack_start_ind <- ind_mat[j, 1] - h
        #   stack_stop_ind <- ind_mat[j, 2] - h
        #   print(paste("j = ", j, ", stack_start_ind = ", stack_start_ind, ", stack_stop_ind = ", stack_stop_ind, sep = ""))
        #   eta_j_old <- eta_stack_old[stack_start_ind : stack_stop_ind]
        #   eta_mat_old <- matrix(eta_stack, nrow = k_n)    # eta_stack_old in matrix form, k_n * p
        #
        #   eta_sum_wo_j <- rowSums(eta_mat_old[, -j, drop = FALSE])
        #   theta_j <- x_mat[, ind_mat[j, 1] : ind_mat[j, 2], drop = F]
        #   theta_inv <- eta_inv_stack[, stack_start_ind : stack_stop_ind]
        #
        #   # print(eta_sum_wo_j)
        #   # compute alpha_j at current iteration
        #   alpha_j <- 1 / 4 * t(theta_j) %*% theta_j %*% eta_j_old + t(theta_j) %*% (y_vec - pi_vec) - mu_1_vec_old - mu_2 * eta_sum_wo_j
        #   # debug log
        #   # print(alpha_j)
        #   # compute the update
        #   eta_j <- theta_inv %*% alpha_j
        #   eta_stack[stack_start_ind : stack_stop_ind] <- eta_j
        #   # print(eta_stack_old)
        #   # print(eta_stack)
        #   print(paste("j = ", j, sep = ""))
        #   Compute_Loss(x_mat, y_vec, delta, eta_stack, mu_1_vec, mu_2, h, k_n, p, 0, oracle_loss = TRUE, print_res = T)
        #   # logit_vec <- x_mat %*% c(delta, eta_stack)
        #   # pi_vec <- exp(logit_vec) / (1 + exp(logit_vec))
        # }
        # update eta_j together
        alpha_eta <- h_eta %*% eta_stack_old + t(x_eta_mat) %*% (y_vec - pi_vec) - t(c_mat) %*% mu_1_vec_old
        eta_stack <- h_eta_inv %*% alpha_eta
        
        # debug log
        # print("After updating eta, loss changes from")
        # Compute_Loss(x_mat, y_vec, delta, eta_stack, mu_1_vec, mu_2, h, k_n, p, 0, oracle_loss = TRUE, print_res = T)
        # print("to")
        # Compute_Loss(x_mat, y_vec, delta, eta_stack, mu_1_vec, mu_2, h, k_n, p, 0, oracle_loss = TRUE, print_res = T)
        
        
        # thresh <- 50
        # for(i in 1 : length(eta_stack)){
        #   if(eta_stack[i] > thresh){
        #     eta_stack[i] <- thresh
        #   }
        #   if(eta_stack[i] < -thresh){
        #     eta_stack[i] <- -thresh
        #   }
        # }
        # step 4. update mu_1
        # each COLUMN for one covariates
        eta_mat <- matrix(eta_stack, nrow = k_n)    # eta_stack in matrix form, k_n * p
        mu_1_vec <- mu_1_vec_old + mu_2 * rowSums(eta_mat)
        
        # debug log
        # print("Updating mu_1_vec from ")
        # print(mu_1_vec_old)
        # print("to")
        # print(mu_1_vec)
        
        # check convergency
        current_iter <- current_iter + 1
        diff1 <- sqrt(sum((delta - delta_old) ^ 2))
        diff2 <- sqrt(sum((eta_stack - eta_stack_old) ^ 2))
        diff <- max(diff1, diff2)
        # debug log
        # print(paste("iter_num = ", current_iter, " diff1 = ", diff1, ", diff2 = ", diff2, sep = ""))
        # print(as.vector(delta))
        logit_vec <- x_mat %*% c(delta, eta_stack)
        loglik <- sum(y_vec * logit_vec - log(1 + exp(logit_vec)))
        loss_p2 <- t(mu_1_vec) %*% rowSums(eta_mat) + mu_2 / 2 * t(rowSums(eta_mat)) %*% rowSums(eta_mat)
        loss <- -loglik + loss_p2
        
        # debug log
        # print(paste("loglik = ", loglik, ", loss_p2 = ", loss_p2, ", loss = ", loss, sep = ""))
        # print(paste("eta row sum: ", paste(rowSums(eta_mat), collapse = " "), sep = ""))
        # Compute_Loss(x_mat, y_vec, delta, eta_stack, mu_1_vec, mu_2, h, k_n, p, 0, oracle_loss = TRUE, print_res = T)
        
        print(paste("iter_num = ", current_iter, " diff1 = ", diff1, ", diff2 = ", diff2, ", loss = ", loss, sep = ""))
        
        if(loss > (loss_old + 1)){
            loss_drop <- FALSE
            if(diff <= tol){
                converge <- TRUE
            }
        }else{
            if(diff <= tol){
                converge <- TRUE
            }
        }
    }
    
    ###### ------------ summary the result ------------
    res <- list(delta = delta,
                eta_stack = eta_stack,
                mu_1_vec = mu_1_vec,
                iter_num = current_iter,
                converge = converge,
                loss_drop = loss_drop
    )
    return(res)
}