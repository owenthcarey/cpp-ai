#include <Eigen/Dense>
#include <LinearRegression.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("LinearRegression learns simple linear relationship", "[linear]") {
    // y = 2*x0 - 3*x1 + 5
    const int n = 200;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        double a = static_cast<double>(i) / n;
        double b = static_cast<double>(n - i) / n;
        X(i, 0) = a;
        X(i, 1) = b;
        y(i) = 2.0 * a - 3.0 * b + 5.0;
    }

    LinearRegression lr;
    lr.train(X, y, 0.5, 1000);

    Eigen::VectorXd preds = lr.predict(X);
    double mse = (preds - y).squaredNorm() / static_cast<double>(n);
    REQUIRE(mse < 1e-6);
}
