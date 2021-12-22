#include <RcppEigen.h>
#include <math.h>
#include "penalty.hpp"
// [[Rcpp::depends(RcppEigen)]]

double SoftThresholding(const double z, const double r) {
/*
     This function is used to perform softthresholding on z at r
     res = z - r    if z > fabs(r)
           z + r    if z < -fabs(r)
           0        o.w.
     Args: z: input value.
           r: thresholding point, use its absolute value.
     Returns: res: softhresholding of z at r
*/
    double holding_point;
    holding_point = fabs(r);
    if(z > holding_point) {
        return(z - holding_point);
    }
    if(z < -holding_point) {
        return(z + holding_point);
    }
    return(0.0);
}


double Penalty_Lasso(const double &input,
                     const Eigen::VectorXd &param,
                     const bool &derivative){
/*
 * This is Lasso penalty on one double input.
 * Args: input: input value
 *       param: lambda = param[0]
 *       derivative: should the derivative be returned instead of the penalized value.
 */
    const double lam = fabs(param[0]);
    double res;
    if(derivative){
        if(input >= 0){
            res = lam;
        }else{
            res = -lam;
        }
    }else{
        res = lam * fabs(input);
    }
    return(res);
}


double Penalty_SCAD(const double &input,
                    const Eigen::VectorXd &param,
                    const bool &derivative){
/*
 * This is SCAD penalty on one double input.
 * Args: input: input value
 *       param: lambda = param[0]
 *              gamma = param[1]
 *       derivative: should the derivative be returned instead of the penalized value.
 */
    const double lam = fabs(param[0]);
    const double gam = fabs(param[1]);
    double res;
    double tmp;
    if(derivative){
            tmp= input;
            if(tmp > lam * gam){
                res = 0;
            }else if(tmp >= lam){
                res = (lam * gam - tmp) / (gam - 1);
            }else if(tmp >= 0){
                res = lam;
            }else if(tmp > -lam){
                res = -lam;
            }else if(tmp > -lam * gam){
                res = (-gam * lam - tmp) / (gam - 1);
            }else{
                res = 0;
            }
    }else{
            tmp = input;
            if(tmp> lam * gam){
                res = (gam + 1) * lam * lam / 2;
            }else if(tmp >= lam){
                res = (-tmp * tmp / 2 + gam * lam * tmp) / (gam - 1) - lam * lam / 2 / (gam - 1);
            }else if(tmp > -lam){
                res = lam * fabs(tmp);
            }else if(tmp > -lam * gam){
                res = (-tmp * tmp / 2 - gam * lam * tmp) / (gam - 1) - lam * lam / 2 / (gam - 1);
            }else{
                res = (gam + 1) * lam * lam / 2;
            }
    }
    return(res);
}


double Penalty_MCP(const double &input,
                   const Eigen::VectorXd &param,
                   const bool &derivative){
    /*
     * This is MCP penalty on one double input.
     * Args: input: input value
     *       param: lambda = param[0]
     *              gamma = param[1]
     *       derivative: should the derivative be returned instead of the penalized value.
     */
    const double lam = fabs(param[0]);
    const double gam = fabs(param[1]);
    double res;
    double tmp;
    if(derivative){
            tmp= input;
            if(tmp > gam * lam){
                res = 0;
            }else if(tmp >= 0){
                res = lam - tmp / gam;
            }else if(tmp > -lam * gam){
                res = -lam - tmp / gam;
            }else{
                res = 0;
            }
    }else{
            tmp = input;
            if(tmp > gam * lam){
                res = gam * lam * lam / 2;
            }else if(tmp >= 0){
                res = lam * tmp - tmp * tmp / 2 / gam;
            }else if(tmp > -gam * lam){
                res = -lam * tmp - tmp * tmp / 2 / gam;
            }else{
                res = gam * lam * lam / 2;
            }
    }
    return(res);
}





