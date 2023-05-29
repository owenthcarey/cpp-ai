//
// Created by owenthcarey on 5/28/23.
//

#include "LinearRegression.h"

LinearRegression::LinearRegression() : bias(0) {}

void LinearRegression::train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                             double learning_rate, int iterations) {
    int n_samples = X.rows();
    int n_features = X.cols();

    // Initialize weights
    weights = Eigen::VectorXd::Zero(n_features);

    for (int i = 0; i < iterations; i++) {
        Eigen::VectorXd y_pred = X * weights + Eigen::VectorXd::Constant(n_samples, bias);
        Eigen::VectorXd diff = y_pred - y;

        weights -= (learning_rate / n_samples) * (X.transpose() * diff);
        bias -= (learning_rate / n_samples) * diff.sum();
    }
}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X) const {
    return X * weights + Eigen::VectorXd::Constant(X.rows(), bias);
}
