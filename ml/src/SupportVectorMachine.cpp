//
// Created by Owen Carey on 8/4/23.
//

#include "../include/SupportVectorMachine.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

// Default constructor
SupportVectorMachine::SupportVectorMachine() : bias(0.0) {}

// Copy constructor
SupportVectorMachine::SupportVectorMachine(const SupportVectorMachine& other)
    : weights(other.weights), bias(other.bias) {}

// Copy assignment operator
SupportVectorMachine& SupportVectorMachine::operator=(const SupportVectorMachine& other) {
    if (this != &other) {
        weights = other.weights;
        bias = other.bias;
    }
    return *this;
}

// Move constructor
SupportVectorMachine::SupportVectorMachine(SupportVectorMachine&& other) noexcept
    : weights(std::move(other.weights)), bias(other.bias) {}

// Move assignment operator
SupportVectorMachine& SupportVectorMachine::operator=(SupportVectorMachine&& other) noexcept {
    if (this != &other) {
        weights = std::move(other.weights);
        bias = other.bias;
    }
    return *this;
}

// Destructor
SupportVectorMachine::~SupportVectorMachine() {}

// Train linear SVM using hinge loss with L2 regularization
// y is expected in {0,1}; internally mapped to {-1,+1}
void SupportVectorMachine::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                 double learning_rate, int iterations, double C) {
    const int n_samples = static_cast<int>(X.rows());
    const int n_features = static_cast<int>(X.cols());
    weights = Eigen::VectorXd::Zero(n_features);
    bias = 0.0;

    // Map labels to {-1,+1}
    Eigen::VectorXd y_signed(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        y_signed(i) = (y(i) >= 0.5) ? 1.0 : -1.0;
    }

    for (int it = 0; it < iterations; ++it) {
        // Compute margins: y_i * (w^T x_i + b)
        Eigen::VectorXd margins = (X * weights).array() + bias;
        margins = (margins.array() * y_signed.array()).matrix();

        // Indicator for hinge loss active (margin < 1)
        Eigen::ArrayXd active = (margins.array() < 1.0).cast<double>();

        // Gradient for weights: w - C * sum_i [active_i * y_i * x_i]
        Eigen::VectorXd grad_w = weights;
        if (active.any()) {
            // Accumulate X^T * (active * y)
            Eigen::VectorXd coeff = (active * y_signed.array()).matrix();
            grad_w -= C * (X.transpose() * coeff);
        }

        // Gradient for bias: -C * sum_i [active_i * y_i]
        double grad_b = 0.0;
        if (active.any()) {
            grad_b = -C * (active * y_signed.array()).sum();
        }

        // Update parameters with learning rate scaled by 1/n for stability
        double scale = learning_rate / static_cast<double>(n_samples);
        weights -= scale * grad_w;
        bias -= scale * grad_b;
    }
}

// Predict labels in {0,1}
Eigen::VectorXd SupportVectorMachine::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd scores = X * weights + Eigen::VectorXd::Constant(X.rows(), bias);
    Eigen::VectorXd preds(scores.size());
    for (int i = 0; i < scores.size(); ++i) preds(i) = scores(i) >= 0.0 ? 1.0 : 0.0;
    return preds;
}

// Split data into train/val/test with shuffling
void SupportVectorMachine::splitData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
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
