//
// Created by owenthcarey on 5/28/23.
//

#include "LinearRegression.h"

// Constructor - initializes bias to 0
LinearRegression::LinearRegression() : bias(0) {}

// train function - trains the linear regression model
// X: Feature Matrix
// y: Target Vector
// learning_rate: Learning rate for gradient descent
// iterations: Number of iterations for training
void LinearRegression::train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                             double learning_rate, int iterations) {
    // Number of samples
    int n_samples = X.rows();
    // Number of features
    int n_features = X.cols();
    // Initialize weights to 0
    weights = Eigen::VectorXd::Zero(n_features);
    // Gradient Descent
    for (int i = 0; i < iterations; i++) {
        // Predicted values
        Eigen::VectorXd y_pred = X * weights + Eigen::VectorXd::Constant(n_samples, bias);
        // Difference between predicted and actual values
        Eigen::VectorXd diff = y_pred - y;
        // Update weights and bias
        weights -= (learning_rate / n_samples) * (X.transpose() * diff);
        bias -= (learning_rate / n_samples) * diff.sum();
    }
}

// predict function - makes predictions based on the trained model
// X: Feature Matrix
Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X) const {
    return X * weights + Eigen::VectorXd::Constant(X.rows(), bias);
}
