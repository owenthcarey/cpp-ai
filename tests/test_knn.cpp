#include <Eigen/Dense>
#include <KNearestNeighbors.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("KNearestNeighbors classifies simple XOR-like clusters", "[knn]") {
    // Create two clusters in 2D: class 0 around (-1, -1), class 1 around (1, 1)
    const int points_per_class = 50;
    const int n = points_per_class * 2;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXd y(n);

    // Class 0
    for (int i = 0; i < points_per_class; ++i) {
        X(i, 0) = -1.0 + 0.1 * (i % 5);
        X(i, 1) = -1.0 + 0.1 * (i / 5);
        y(i) = 0.0;
    }
    // Class 1
    for (int i = 0; i < points_per_class; ++i) {
        int idx = points_per_class + i;
        X(idx, 0) = 1.0 + 0.1 * (i % 5);
        X(idx, 1) = 1.0 + 0.1 * (i / 5);
        y(idx) = 1.0;
    }

    KNearestNeighbors knn;
    knn.train(X, y, 3);

    Eigen::VectorXd preds = knn.predict(X);
    double acc = (preds.array() == y.array()).cast<double>().mean();
    REQUIRE(acc >= 0.98);
}
