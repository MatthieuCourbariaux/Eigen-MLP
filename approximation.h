/*****************************************************************
Matrix product approximation
Written by Matthieu Courbariaux in 2015
*******************************************************************/

#ifndef MPA
#define MPA

#include <Eigen/Core>
using Eigen::MatrixXf;

// Train the NN using gradient descent
MatrixXf prod_approx(MatrixXf A,MatrixXf B);

#endif