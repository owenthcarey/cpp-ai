//
// Created by Owen Carey on 8/5/23.
//

#include "../include/KMeansClustering.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// Default constructor
KMeansClustering::KMeansClustering() : k(0) {}

// Copy constructor
KMeansClustering::KMeansClustering(const KMeansClustering& other)
    : centroids(other.centroids), k(other.k) {}

// Copy assignment operator
KMeansClustering& KMeansClustering::operator=(const KMeansClustering& other) {
    if (this != &other) {
        centroids = other.centroids;
        k = other.k;
    }
    return *this;
}

// Move constructor
KMeansClustering::KMeansClustering(KMeansClustering&& other) noexcept
    : centroids(std::move(other.centroids)), k(other.k) {}

// Move assignment operator
KMeansClustering& KMeansClustering::operator=(KMeansClustering&& other) noexcept {
    if (this != &other) {
        centroids = std::move(other.centroids);
        k = other.k;
    }
    return *this;
}

// Destructor
KMeansClustering::~KMeansClustering() {}

static inline int argmin_rowwise_distance(const Eigen::RowVectorXd& x,
                                          const Eigen::MatrixXd& centroids) {
    int best_idx = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (int c = 0; c < centroids.rows(); ++c) {
        double d2 = (x - centroids.row(c)).squaredNorm();
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_idx = c;
        }
    }
    return best_idx;
}

void KMeansClustering::train(const Eigen::MatrixXd& X, int num_clusters, int max_iters) {
    const int n_samples = static_cast<int>(X.rows());
    const int n_features = static_cast<int>(X.cols());
    k = std::max(1, std::min(num_clusters, n_samples));
    if (k == 0) {
        centroids.resize(0, 0);
        return;
    }

    // Initialize centroids by sampling k unique random points
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    centroids.resize(k, n_features);
    for (int c = 0; c < k; ++c) {
        centroids.row(c) = X.row(indices[c]);
    }

    Eigen::VectorXi assignments = Eigen::VectorXi::Zero(n_samples);
    for (int iter = 0; iter < max_iters; ++iter) {
        // Assign step
        bool changed = false;
        for (int i = 0; i < n_samples; ++i) {
            int new_label = argmin_rowwise_distance(X.row(i), centroids);
            if (new_label != assignments(i)) {
                assignments(i) = new_label;
                changed = true;
            }
        }

        // Update step
        Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(k, n_features);
        std::vector<int> counts(k, 0);
        for (int i = 0; i < n_samples; ++i) {
            int lbl = assignments(i);
            new_centroids.row(lbl) += X.row(i);
            counts[lbl] += 1;
        }
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                new_centroids.row(c) /= static_cast<double>(counts[c]);
            } else {
                // Reinitialize empty cluster to a random data point
                int idx = std::uniform_int_distribution<int>(0, n_samples - 1)(gen);
                new_centroids.row(c) = X.row(idx);
            }
        }

        // Check convergence
        if (!changed || (new_centroids - centroids).norm() < 1e-9) {
            centroids = std::move(new_centroids);
            break;
        }
        centroids = std::move(new_centroids);
    }
}

Eigen::VectorXd KMeansClustering::predict(const Eigen::MatrixXd& X) const {
    const int n_samples = static_cast<int>(X.rows());
    Eigen::VectorXd labels(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        labels(i) = static_cast<double>(argmin_rowwise_distance(X.row(i), centroids));
    }
    return labels;
}
