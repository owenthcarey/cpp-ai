//
// Created by owenthcarey on 5/28/23.
//

#include "LinearRegression.h"

// Default constructor that initializes bias to 0.
LinearRegression::LinearRegression() : bias(0) {}

// Trains the linear-regression model.
// Parameters:
// - X: feature matrix.
// - y: target vector.
// - learning_rate: the learning rate for gradient-descent training.
// - iterations: the number of iterations for gradient-descent training.
void LinearRegression::train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                             double learning_rate, int iterations) {
    int n_samples = X.rows();
    int n_features = X.cols();
    weights = Eigen::VectorXd::Zero(n_features);
    for (int i = 0; i < iterations; i++) {
        Eigen::VectorXd y_pred =
                X * weights + Eigen::VectorXd::Constant(n_samples, bias);
        Eigen::VectorXd diff = y_pred - y;
        weights -= (learning_rate / n_samples) * (X.transpose() * diff);
        bias -= (learning_rate / n_samples) * diff.sum();
    }
}

// Makes predictions based on the trained model.
// Parameter:
// - X: feature matrix.
// Returns:
// - Predicted output vector.
Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X) const {
    return X * weights + Eigen::VectorXd::Constant(X.rows(), bias);
}
