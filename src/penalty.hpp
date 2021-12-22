#ifndef PENALTY_H
#define PENALTY_H

/* header file contents go here */

double SoftThresholding(const double z, const double r);

double Penalty_Lasso(const double &input,
                     const Eigen::VectorXd &param,
                     const bool &derivative);

double Penalty_SCAD(const double &input,
                    const Eigen::VectorXd &param,
                    const bool &derivative);

double Penalty_MCP(const double &input,
                   const Eigen::VectorXd &param,
                   const bool &derivative);



#endif /* PENALTY_H */




