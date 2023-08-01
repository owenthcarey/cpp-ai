//
// Created by owenthcarey on 5/28/23.
//

#include "../include/LinearRegression.h"

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

// Splits data into training, validation, and test sets.
void
LinearRegression::splitData(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                            Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train,
                            Eigen::MatrixXd &X_val, Eigen::VectorXd &y_val,
                            Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test,
                            double train_size, double val_size) {
    int n_samples = X.rows();
    int train_samples = int(n_samples * train_size);
    int val_samples = int(n_samples * val_size);

    // Generate a vector with indices and shuffle it
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Create shuffled copies of X and y
    Eigen::MatrixXd X_shuffled(n_samples, X.cols());
    Eigen::VectorXd y_shuffled(n_samples);
    for (int i = 0; i < n_samples; i++) {
        X_shuffled.row(i) = X.row(indices[i]);
        y_shuffled(i) = y(indices[i]);
    }

    // Split the shuffled data
    X_train = X_shuffled.topRows(train_samples);
    y_train = y_shuffled.head(train_samples);
    X_val = X_shuffled.middleRows(train_samples, val_samples);
    y_val = y_shuffled.segment(train_samples, val_samples);
    X_test = X_shuffled.bottomRows(n_samples - train_samples - val_samples);
    y_test = y_shuffled.tail(n_samples - train_samples - val_samples);
}
