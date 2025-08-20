//
// Created by Owen Carey on 8/3/23.
//

#ifndef CPP_AI_KNEARESTNEIGHBORS_H
#define CPP_AI_KNEARESTNEIGHBORS_H

#include <Eigen/Dense>
#include <random>

class KNearestNeighbors {
  private:
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;
    int k;

  public:
    // Default constructor
    KNearestNeighbors();

    // Copy constructor
    KNearestNeighbors(const KNearestNeighbors& other);

    // Copy assignment operator
    KNearestNeighbors& operator=(const KNearestNeighbors& other);

    // Move constructor
    KNearestNeighbors(KNearestNeighbors&& other) noexcept;

    // Move assignment operator
    KNearestNeighbors& operator=(KNearestNeighbors&& other) noexcept;

    // Destructor
    ~KNearestNeighbors();

    // Train by memorizing the dataset and setting k
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int k_neighbors);

    // Predict class labels (0/1) using majority vote among k nearest neighbors
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;

    // Utility: split data into train/val/test with shuffling
    static void splitData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                          Eigen::MatrixXd& X_out_train, Eigen::VectorXd& y_out_train,
                          Eigen::MatrixXd& X_out_val, Eigen::VectorXd& y_out_val,
                          Eigen::MatrixXd& X_out_test, Eigen::VectorXd& y_out_test,
                          double train_size = 0.8, double val_size = 0.1);
};

#endif // CPP_AI_KNEARESTNEIGHBORS_H
