//
// Created by owenthcarey on 5/28/23.
//

#ifndef CPP_AI_LINEARREGRESSION_H
#define CPP_AI_LINEARREGRESSION_H

#include <Eigen/Dense>

class LinearRegression {
private:
    Eigen::VectorXd weights;
    double bias;

public:
    LinearRegression();
    void train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
               double learning_rate, int iterations);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
};

#endif //CPP_AI_LINEARREGRESSION_H
