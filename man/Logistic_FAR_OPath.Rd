% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/opath_solver.R
\name{Logistic_FAR_OPath}
\alias{Logistic_FAR_OPath}
\title{Finds the solution path of logistic functional additive regression with log-contrast constrain.}
\usage{
Logistic_FAR_OPath(
  y_vec,
  x_mat,
  h,
  kn,
  p,
  p_type,
  p_param,
  lambda_seq,
  lambda_length,
  min_lambda_ratio = 0.01,
  mu2,
  a,
  bj_vec,
  cj_vec,
  rj_vec,
  delta_init,
  eta_stack_init,
  mu1_init,
  tol = 10^(-6),
  max_iter = 500,
  verbose = 0,
  svd_thresh = 10^(-7)
)
}
\arguments{
\item{y_vec}{response vector, 0 for control, 1 for case.
n = length(y_vec) is the number of observations.}

\item{x_mat}{covariate matrix, consists of two parts.
dim(x_mat) = (n, h + p * kn)
First h columns are for demographical covariates(can include an intercept term)
Rest columns are for p functional covariates, each being represented by a set of basis functions resulting kn covariates.}

\item{h, kn, p}{dimension information for the dataset(\code{x_mat}).}

\item{p_type}{an character variable indicating different types of the penalty}

\item{p_param}{numerical vector for the penalty function.
\code{p_param[1]} store sthe lambda value and will be provided by \code{lambda_seq}.}

\item{lambda_seq}{a non-negative sequence of lambda, along which the solution path is searched.
It is RECOMMENED to not supply this parameter and let the function itself determines
it from the given data.}

\item{lambda_length}{length of the lambda sequence when computing \code{lambda_seq}.
If \code{lambda_seq} is provided, then of course \code{lambda_length = length(lambda_seq)}.}

\item{mu2}{quadratic term in the ADMM algorithm}

\item{a, bj_vec, cj_vec, rj_vec}{parameters for the algorithm. See Algorithm_Details.pdf
for more information.}

\item{delta_init, eta_stack_init, mu1_init}{initial values for the algorithm.}

\item{tol, max_iter}{convergence tolerance and max number of iteration of the algorithm.}

\item{verbose}{integer, indicating level of information to be printed during computation, currently supports:
always: some info if something went wrong, e.g. when no penalty function is matched
1: information about the start and stop of the iteration
2. How the loss value is changed during each iteration}

\item{svd_thresh}{a small value for threashing the singular value vectors.}

\item{min_lam_ratio:}{\code{min(lambda_seq) / max{lambda_seq}}. This function uses this
parameter to determine the minimal value of \code{lambda_seq}. If \code{p > n}, then it
is recommended to set this no smaller than 0.01 (sometimes even 0.05), otherwise you can
set it to 0.001 or even smaller.}
}
\value{
A list containing the solution path of \code{delta}, \code{eta_stack}, \code{mu1}
and some computation information such as convergency, iteration number and the lambda
sequence of this solution path.
}
\description{
\code{Logistic_FAR_OPath} finds the solution path of logistic functional additive regression
with log-contrast constrain. It will use within-group orthonormalization to
standardize the data before the real computation.
}
