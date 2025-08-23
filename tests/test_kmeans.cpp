#include <Eigen/Dense>
#include <KMeansClustering.h>
#include <catch2/catch_test_macros.hpp>
#include <limits>

static double min_distance_to_target(const Eigen::MatrixXd& C, const Eigen::Vector2d& target) {
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < C.rows(); ++i) {
        best = std::min(best, (C.row(i).transpose() - target).norm());
    }
    return best;
}

TEST_CASE("KMeansClustering finds two well-separated cluster centers", "[kmeans]") {
    const int points_per_class = 50;
    const int n = points_per_class * 2;
    Eigen::MatrixXd X(n, 2);

    // Deterministic grid clusters around (-1, -1) and (1, 1)
    for (int i = 0; i < points_per_class; ++i) {
        X(i, 0) = -1.0 + 0.1 * (i % 5);
        X(i, 1) = -1.0 + 0.1 * (i / 5);
    }
    for (int i = 0; i < points_per_class; ++i) {
        int idx = points_per_class + i;
        X(idx, 0) = 1.0 + 0.1 * (i % 5);
        X(idx, 1) = 1.0 + 0.1 * (i / 5);
    }

    KMeansClustering km;
    km.train(X, 2, 100);

    const Eigen::MatrixXd& C = km.getCentroids();
    REQUIRE(C.rows() == 2);
    REQUIRE(C.cols() == 2);

    // Expected means of the two deterministic grids
    Eigen::Vector2d center_neg(-0.8, -0.55);
    Eigen::Vector2d center_pos(1.2, 1.45);

    double d_neg = min_distance_to_target(C, center_neg);
    double d_pos = min_distance_to_target(C, center_pos);
    REQUIRE(d_neg < 0.25);
    REQUIRE(d_pos < 0.25);
}
