#include<RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;
using namespace Rcpp;

// [[Rcpp::export]]
double mycppsum(const Eigen::VectorXd& invec){
    double res = invec.sum();
    return(res);
}
