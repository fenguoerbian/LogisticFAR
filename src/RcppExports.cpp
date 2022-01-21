// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Logistic_FAR_FLiRTI_Solver_Core
Rcpp::List Logistic_FAR_FLiRTI_Solver_Core(const Eigen::VectorXd& y_vec, const Eigen::MatrixXd& x_mat, const int& h, const int& kn, const int& p, const char& p_type, const Eigen::VectorXd& p_param, const double& mu2, const double& a, const Eigen::VectorXd& bj_vec, const Eigen::VectorXd& cj_vec, const Eigen::VectorXd& rj_vec, const Eigen::VectorXd& weight_vec, const double& tol, const int& max_iter, const Eigen::VectorXd& relax_vec, const Eigen::MatrixXd& hd_mat, const Eigen::MatrixXd& hd_inv, const Eigen::VectorXd& delta_init, const Eigen::VectorXd& eta_stack_init, const Eigen::VectorXd& mu1_init);
RcppExport SEXP _LogisticFAR_Logistic_FAR_FLiRTI_Solver_Core(SEXP y_vecSEXP, SEXP x_matSEXP, SEXP hSEXP, SEXP knSEXP, SEXP pSEXP, SEXP p_typeSEXP, SEXP p_paramSEXP, SEXP mu2SEXP, SEXP aSEXP, SEXP bj_vecSEXP, SEXP cj_vecSEXP, SEXP rj_vecSEXP, SEXP weight_vecSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP relax_vecSEXP, SEXP hd_matSEXP, SEXP hd_invSEXP, SEXP delta_initSEXP, SEXP eta_stack_initSEXP, SEXP mu1_initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y_vec(y_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type x_mat(x_matSEXP);
    Rcpp::traits::input_parameter< const int& >::type h(hSEXP);
    Rcpp::traits::input_parameter< const int& >::type kn(knSEXP);
    Rcpp::traits::input_parameter< const int& >::type p(pSEXP);
    Rcpp::traits::input_parameter< const char& >::type p_type(p_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type p_param(p_paramSEXP);
    Rcpp::traits::input_parameter< const double& >::type mu2(mu2SEXP);
    Rcpp::traits::input_parameter< const double& >::type a(aSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type bj_vec(bj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type cj_vec(cj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type rj_vec(rj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type weight_vec(weight_vecSEXP);
    Rcpp::traits::input_parameter< const double& >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const int& >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type relax_vec(relax_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type hd_mat(hd_matSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type hd_inv(hd_invSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type delta_init(delta_initSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type eta_stack_init(eta_stack_initSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type mu1_init(mu1_initSEXP);
    rcpp_result_gen = Rcpp::wrap(Logistic_FAR_FLiRTI_Solver_Core(y_vec, x_mat, h, kn, p, p_type, p_param, mu2, a, bj_vec, cj_vec, rj_vec, weight_vec, tol, max_iter, relax_vec, hd_mat, hd_inv, delta_init, eta_stack_init, mu1_init));
    return rcpp_result_gen;
END_RCPP
}
// Logistic_FAR_Ortho_Solver_Core
Rcpp::List Logistic_FAR_Ortho_Solver_Core(const Eigen::VectorXd& y_vec, const Eigen::MatrixXd& x_mat, const int& h, const int& kn, const int& p, const char& p_type, const Eigen::VectorXd& p_param, const double& mu2, const double& a, const Eigen::VectorXd& bj_vec, const Eigen::VectorXd& cj_vec, const Eigen::VectorXd& rj_vec, const Eigen::VectorXd& weight_vec, const double& tol, const int& max_iter, const Eigen::VectorXd& relax_vec, const Eigen::MatrixXd& t_mat_stack, const Eigen::VectorXd& svd_vec_stack, const Eigen::VectorXd& start_id_vec, const Eigen::MatrixXd& hd_mat, const Eigen::MatrixXd& hd_inv, const Eigen::VectorXd& delta_init, const Eigen::VectorXd& eta_stack_init, const Eigen::VectorXd& mu1_init);
RcppExport SEXP _LogisticFAR_Logistic_FAR_Ortho_Solver_Core(SEXP y_vecSEXP, SEXP x_matSEXP, SEXP hSEXP, SEXP knSEXP, SEXP pSEXP, SEXP p_typeSEXP, SEXP p_paramSEXP, SEXP mu2SEXP, SEXP aSEXP, SEXP bj_vecSEXP, SEXP cj_vecSEXP, SEXP rj_vecSEXP, SEXP weight_vecSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP relax_vecSEXP, SEXP t_mat_stackSEXP, SEXP svd_vec_stackSEXP, SEXP start_id_vecSEXP, SEXP hd_matSEXP, SEXP hd_invSEXP, SEXP delta_initSEXP, SEXP eta_stack_initSEXP, SEXP mu1_initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y_vec(y_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type x_mat(x_matSEXP);
    Rcpp::traits::input_parameter< const int& >::type h(hSEXP);
    Rcpp::traits::input_parameter< const int& >::type kn(knSEXP);
    Rcpp::traits::input_parameter< const int& >::type p(pSEXP);
    Rcpp::traits::input_parameter< const char& >::type p_type(p_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type p_param(p_paramSEXP);
    Rcpp::traits::input_parameter< const double& >::type mu2(mu2SEXP);
    Rcpp::traits::input_parameter< const double& >::type a(aSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type bj_vec(bj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type cj_vec(cj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type rj_vec(rj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type weight_vec(weight_vecSEXP);
    Rcpp::traits::input_parameter< const double& >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const int& >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type relax_vec(relax_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type t_mat_stack(t_mat_stackSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type svd_vec_stack(svd_vec_stackSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type start_id_vec(start_id_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type hd_mat(hd_matSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type hd_inv(hd_invSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type delta_init(delta_initSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type eta_stack_init(eta_stack_initSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type mu1_init(mu1_initSEXP);
    rcpp_result_gen = Rcpp::wrap(Logistic_FAR_Ortho_Solver_Core(y_vec, x_mat, h, kn, p, p_type, p_param, mu2, a, bj_vec, cj_vec, rj_vec, weight_vec, tol, max_iter, relax_vec, t_mat_stack, svd_vec_stack, start_id_vec, hd_mat, hd_inv, delta_init, eta_stack_init, mu1_init));
    return rcpp_result_gen;
END_RCPP
}
// Logistic_FAR_Solver_Core
Rcpp::List Logistic_FAR_Solver_Core(const Eigen::VectorXd& y_vec, const Eigen::MatrixXd& x_mat, const int& h, const int& kn, const int& p, const char& p_type, const Eigen::VectorXd& p_param, const double& mu2, const double& a, const Eigen::VectorXd& bj_vec, const Eigen::VectorXd& cj_vec, const Eigen::VectorXd& rj_vec, const Eigen::VectorXd& weight_vec, const double& tol, const int& max_iter, const Eigen::VectorXd& relax_vec, const Eigen::MatrixXd& hd_mat, const Eigen::MatrixXd& hd_inv, const Eigen::VectorXd& delta_init, const Eigen::VectorXd& eta_stack_init, const Eigen::VectorXd& mu1_init);
RcppExport SEXP _LogisticFAR_Logistic_FAR_Solver_Core(SEXP y_vecSEXP, SEXP x_matSEXP, SEXP hSEXP, SEXP knSEXP, SEXP pSEXP, SEXP p_typeSEXP, SEXP p_paramSEXP, SEXP mu2SEXP, SEXP aSEXP, SEXP bj_vecSEXP, SEXP cj_vecSEXP, SEXP rj_vecSEXP, SEXP weight_vecSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP relax_vecSEXP, SEXP hd_matSEXP, SEXP hd_invSEXP, SEXP delta_initSEXP, SEXP eta_stack_initSEXP, SEXP mu1_initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y_vec(y_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type x_mat(x_matSEXP);
    Rcpp::traits::input_parameter< const int& >::type h(hSEXP);
    Rcpp::traits::input_parameter< const int& >::type kn(knSEXP);
    Rcpp::traits::input_parameter< const int& >::type p(pSEXP);
    Rcpp::traits::input_parameter< const char& >::type p_type(p_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type p_param(p_paramSEXP);
    Rcpp::traits::input_parameter< const double& >::type mu2(mu2SEXP);
    Rcpp::traits::input_parameter< const double& >::type a(aSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type bj_vec(bj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type cj_vec(cj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type rj_vec(rj_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type weight_vec(weight_vecSEXP);
    Rcpp::traits::input_parameter< const double& >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const int& >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type relax_vec(relax_vecSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type hd_mat(hd_matSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type hd_inv(hd_invSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type delta_init(delta_initSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type eta_stack_init(eta_stack_initSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type mu1_init(mu1_initSEXP);
    rcpp_result_gen = Rcpp::wrap(Logistic_FAR_Solver_Core(y_vec, x_mat, h, kn, p, p_type, p_param, mu2, a, bj_vec, cj_vec, rj_vec, weight_vec, tol, max_iter, relax_vec, hd_mat, hd_inv, delta_init, eta_stack_init, mu1_init));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_LogisticFAR_Logistic_FAR_FLiRTI_Solver_Core", (DL_FUNC) &_LogisticFAR_Logistic_FAR_FLiRTI_Solver_Core, 21},
    {"_LogisticFAR_Logistic_FAR_Ortho_Solver_Core", (DL_FUNC) &_LogisticFAR_Logistic_FAR_Ortho_Solver_Core, 24},
    {"_LogisticFAR_Logistic_FAR_Solver_Core", (DL_FUNC) &_LogisticFAR_Logistic_FAR_Solver_Core, 21},
    {NULL, NULL, 0}
};

RcppExport void R_init_LogisticFAR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
