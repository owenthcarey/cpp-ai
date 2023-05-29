//
// Created by owenthcarey on 5/28/23.
//

#ifndef CPP_AI_LINEARREGRESSION_H
#define CPP_AI_LINEARREGRESSION_H


#include <Eigen/Dense>

// LinearRegression Class - implements a simple linear regression model
class LinearRegression {
private:
    // Vector of weights (coefficients)
    Eigen::VectorXd weights;
    // Scalar bias (intercept)
    double bias;

public:
    // Constructor
    LinearRegression();
    // Function to train the model
    void train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
               double learning_rate, int iterations);
    // Function to make predictions
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
};


#endif //CPP_AI_LINEARREGRESSION_H
