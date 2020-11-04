#' @export
myfoo <- function(x){
    return(mycppsum(x))
}

#' @useDynLib LogisticFAR
#' @importFrom Rcpp sourceCpp
