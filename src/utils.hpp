#ifndef UTILITY_FUNCTION_H
#define UTILITY_FUNCTION_H

/* header file contents go here */

Eigen::VectorXd Compute_Pi_Vec(const Eigen::MatrixXd &x_mat,
                               const Eigen::VectorXd &delta,
                               const Eigen::VectorXd &eta_stack);

Eigen::VectorXd Rowsum_wo_j(const Eigen::VectorXd &eta_stack, const int &j,
                            const int &kn, const int &p);

Eigen::VectorXd Rowsum(const Eigen::VectorXd &eta_stack, const int &kn, const int &p);

double Compute_Loss_Cpp(const Eigen::MatrixXd &x_mat, const Eigen::VectorXd &y_vec,
                    const Eigen::VectorXd &delta_vec, const Eigen::VectorXd &eta_stack_vec,
                    const Eigen::VectorXd &mu1_vec, const double &mu2,
                    const double &h, const double &kn, const double &p,
                    const char &p_type, const Eigen::VectorXd &p_param,
                    const double &a, const Eigen::VectorXd &bj_vec, const Eigen::VectorXd &cj_vec, const Eigen::VectorXd &rj_vec,
                    const bool &oracle_loss, const bool &print_res);



#endif /* UTILITY_FUNCTION_H */




