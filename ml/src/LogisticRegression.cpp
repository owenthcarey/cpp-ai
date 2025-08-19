//
// Created by Owen Carey on 8/2/23.
//

#include "../include/LogisticRegression.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

// Default constructor
LogisticRegression::LogisticRegression() : bias(0) {}

// Copy constructor
LogisticRegression::LogisticRegression(const LogisticRegression& other)
    : weights(other.weights), bias(other.bias) {}

// Copy assignment operator
LogisticRegression& LogisticRegression::operator=(const LogisticRegression& other) {
    if (this != &other) {
        weights = other.weights;
        bias = other.bias;
    }
    return *this;
}

// Move constructor
LogisticRegression::LogisticRegression(LogisticRegression&& other) noexcept
    : weights(std::move(other.weights)), bias(other.bias) {}

// Move assignment operator
LogisticRegression& LogisticRegression::operator=(LogisticRegression&& other) noexcept {
    if (this != &other) {
        weights = std::move(other.weights);
        bias = other.bias;
    }
    return *this;
}

// Destructor
LogisticRegression::~LogisticRegression() {}

// Train using gradient descent on logistic loss
void LogisticRegression::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                               double learning_rate, int iterations) {
    int n_samples = X.rows();
    int n_features = X.cols();
    weights = Eigen::VectorXd::Zero(n_features);
    bias = 0.0;

    for (int i = 0; i < iterations; i++) {
        Eigen::VectorXd linear_output = X * weights + Eigen::VectorXd::Constant(n_samples, bias);
        Eigen::ArrayXd probabilities = 1.0 / (1.0 + (-linear_output.array()).exp());
        Eigen::VectorXd diff = probabilities.matrix() - y; // y should be in {0,1}

        // Gradients
        Eigen::VectorXd grad_w = (X.transpose() * diff) / static_cast<double>(n_samples);
        double grad_b = diff.sum() / static_cast<double>(n_samples);

        // Update
        weights -= learning_rate * grad_w;
        bias -= learning_rate * grad_b;
    }
}

// Predict probabilities P(y=1|x)
Eigen::VectorXd LogisticRegression::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd linear_output = X * weights + Eigen::VectorXd::Constant(X.rows(), bias);
    Eigen::ArrayXd probabilities = 1.0 / (1.0 + (-linear_output.array()).exp());
    return probabilities.matrix();
}

// Split data into train/val/test, with shuffling
void LogisticRegression::splitData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                   Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train,
                                   Eigen::MatrixXd& X_val, Eigen::VectorXd& y_val,
                                   Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test,
                                   double train_size, double val_size) {
    int n_samples = X.rows();
    int train_samples = static_cast<int>(n_samples * train_size);
    int val_samples = static_cast<int>(n_samples * val_size);

    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Eigen::MatrixXd X_shuffled(n_samples, X.cols());
    Eigen::VectorXd y_shuffled(n_samples);
    for (int i = 0; i < n_samples; i++) {
        X_shuffled.row(i) = X.row(indices[i]);
        y_shuffled(i) = y(indices[i]);
    }

    X_train = X_shuffled.topRows(train_samples);
    y_train = y_shuffled.head(train_samples);
    X_val = X_shuffled.middleRows(train_samples, val_samples);
    y_val = y_shuffled.segment(train_samples, val_samples);
    X_test = X_shuffled.bottomRows(n_samples - train_samples - val_samples);
    y_test = y_shuffled.tail(n_samples - train_samples - val_samples);
}
