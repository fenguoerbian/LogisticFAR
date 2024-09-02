n <- 100
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)

xf <- cbind(x1, x2, x1, x3, x1)
colnames(xf) <- letters[1 : 5]

pi_vec <- plogis(x1)
y1 <- rbinom(n, size = 1, prob = pi_vec)
y2 <- dplyr::if_else(pi_vec < 0.5, 0, 1)

f1 <- glm(y1 ~ x1, family = binomial)
f2 <- glm(y1 ~ xf, family = binomial)
f3 <- glm(y2 ~ x1, family = binomial)
f4 <- glm(y2 ~ xf, family = binomial)

delta_vec1 <- f1$coefficients
delta_vec1[is.na(delta_vec1)] <- 0
delta_vec1

delta_vec2 <- f2$coefficients
delta_vec2[is.na(delta_vec2)] <- 0


subj_vec <- sample(5, size = n, replace = TRUE)
mf1 <- lme4::glmer(y1 ~ x1 + (1 | subj_vec), family = binomial)
mf2 <- lme4::glmer(y1 ~ xf + (1 | subj_vec), family = binomial)
mf3 <- lme4::glmer(y2 ~ x1 + (1 | subj_vec), family = binomial)
mf4 <- lme4::glmer(y2 ~ xf + (1 | subj_vec), family = binomial)

is.null(attr(mf1@pp$X, "col.dropped"))

tmp <- mf2@pptmXwtsp <- mf2@pp

replace_fixef_na <- function(in_model){
    fixef <- in_model@beta
    dropped_col_idx <- attr(in_model@pp$X, "col.dropped")
    if(!is.null(dropped_col_idx)){
        drop_num <- length(dropped_col_idx)
        valid_num <- length(fixef)
        valid_idx <- setdiff(1 : (drop_num + valid_num), dropped_col_idx)
        res <- rep(0, drop_num + valid_num)
        res[valid_idx] <- fixef
    }else{
        res <- fixef
    }

    return(res)
}

replace_fixef_na(mf2)
