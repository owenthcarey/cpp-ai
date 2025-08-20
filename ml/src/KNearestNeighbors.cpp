//
// Created by Owen Carey on 8/3/23.
//

#include "../include/KNearestNeighbors.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

// Default constructor
KNearestNeighbors::KNearestNeighbors() : k(1) {}

// Copy constructor
KNearestNeighbors::KNearestNeighbors(const KNearestNeighbors& other)
    : X_train(other.X_train), y_train(other.y_train), k(other.k) {}

// Copy assignment operator
KNearestNeighbors& KNearestNeighbors::operator=(const KNearestNeighbors& other) {
    if (this != &other) {
        X_train = other.X_train;
        y_train = other.y_train;
        k = other.k;
    }
    return *this;
}

// Move constructor
KNearestNeighbors::KNearestNeighbors(KNearestNeighbors&& other) noexcept
    : X_train(std::move(other.X_train)), y_train(std::move(other.y_train)), k(other.k) {}

// Move assignment operator
KNearestNeighbors& KNearestNeighbors::operator=(KNearestNeighbors&& other) noexcept {
    if (this != &other) {
        X_train = std::move(other.X_train);
        y_train = std::move(other.y_train);
        k = other.k;
    }
    return *this;
}

// Destructor
KNearestNeighbors::~KNearestNeighbors() {}

void KNearestNeighbors::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                              int k_neighbors) {
    X_train = X;
    y_train = y;
    k = std::max(1, std::min(k_neighbors, static_cast<int>(X_train.rows())));
}

Eigen::VectorXd KNearestNeighbors::predict(const Eigen::MatrixXd& X) const {
    const int num_queries = static_cast<int>(X.rows());
    Eigen::VectorXd predictions(num_queries);

    for (int i = 0; i < num_queries; ++i) {
        // Compute distances to all training points
        std::vector<std::pair<double, int>> distance_index_pairs;
        distance_index_pairs.reserve(static_cast<size_t>(X_train.rows()));

        for (int j = 0; j < X_train.rows(); ++j) {
            // Euclidean distance squared is sufficient for ordering
            double dist2 = (X.row(i) - X_train.row(j)).squaredNorm();
            distance_index_pairs.emplace_back(dist2, j);
        }

        // Partially sort to find k nearest
        const int k_effective = std::min(k, static_cast<int>(distance_index_pairs.size()));
        if (k_effective > 0 && k_effective < static_cast<int>(distance_index_pairs.size())) {
            std::nth_element(distance_index_pairs.begin(),
                             distance_index_pairs.begin() + k_effective,
                             distance_index_pairs.end(),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
        } else {
            // When k_effective equals size or zero, skip nth_element; the entire list is considered
        }

        // Majority vote among k nearest
        int positive_count = 0;
        for (int n = 0; n < k_effective; ++n) {
            int idx = distance_index_pairs[n].second;
            positive_count += (y_train(idx) >= 0.5) ? 1 : 0;
        }

        predictions(i) = (positive_count > k_effective / 2) ? 1.0 : 0.0;
    }

    return predictions;
}

void KNearestNeighbors::splitData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                  Eigen::MatrixXd& X_out_train, Eigen::VectorXd& y_out_train,
                                  Eigen::MatrixXd& X_out_val, Eigen::VectorXd& y_out_val,
                                  Eigen::MatrixXd& X_out_test, Eigen::VectorXd& y_out_test,
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

    X_out_train = X_shuffled.topRows(train_samples);
    y_out_train = y_shuffled.head(train_samples);
    X_out_val = X_shuffled.middleRows(train_samples, val_samples);
    y_out_val = y_shuffled.segment(train_samples, val_samples);
    X_out_test = X_shuffled.bottomRows(n_samples - train_samples - val_samples);
    y_out_test = y_shuffled.tail(n_samples - train_samples - val_samples);
}
