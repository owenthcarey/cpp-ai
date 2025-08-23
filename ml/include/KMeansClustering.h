//
// Created by Owen Carey on 8/5/23.
//

#ifndef CPP_AI_KMEANSCLUSTERING_H
#define CPP_AI_KMEANSCLUSTERING_H

#include <Eigen/Dense>
#include <random>

class KMeansClustering {
  private:
    Eigen::MatrixXd centroids;
    int k;

  public:
    // Default constructor
    KMeansClustering();

    // Copy constructor
    KMeansClustering(const KMeansClustering& other);

    // Copy assignment operator
    KMeansClustering& operator=(const KMeansClustering& other);

    // Move constructor
    KMeansClustering(KMeansClustering&& other) noexcept;

    // Move assignment operator
    KMeansClustering& operator=(KMeansClustering&& other) noexcept;

    // Destructor
    ~KMeansClustering();

    // Train KMeans with k clusters and up to max_iters iterations
    void train(const Eigen::MatrixXd& X, int num_clusters, int max_iters = 100);

    // Predict cluster labels (0..k-1) for each row of X
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;

    // Access learned centroids
    const Eigen::MatrixXd& getCentroids() const { return centroids; }
};

#endif // CPP_AI_KMEANSCLUSTERING_H
