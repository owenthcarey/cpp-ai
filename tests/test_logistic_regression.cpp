#include <Eigen/Dense>
#include <LogisticRegression.h>
#include <catch2/catch_test_macros.hpp>

static Eigen::VectorXd round_probs(const Eigen::VectorXd& p) {
    Eigen::VectorXd out(p.size());
    for (int i = 0; i < p.size(); ++i)
        out(i) = p(i) >= 0.5 ? 1.0 : 0.0;
    return out;
}

TEST_CASE("LogisticRegression separates linearly separable data", "[logistic]") {
    // Two features; decision boundary: x0 - x1 >= 0 -> class 1 else 0
    const int n = 300;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        double a = static_cast<double>(i % 50) / 50.0;       // 0..1 cyclic
        double b = static_cast<double>((i * 7) % 50) / 50.0; // pseudo-random-ish
        X(i, 0) = a;
        X(i, 1) = b;
        y(i) = (a - b >= 0.05) ? 1.0 : 0.0; // small margin to avoid ambiguity
    }

    LogisticRegression clf;
    clf.train(X, y, 0.5, 1500);

    Eigen::VectorXd probs = clf.predict(X);
    Eigen::VectorXd preds = round_probs(probs);
    double acc = (preds.array() == y.array()).cast<double>().mean();
    REQUIRE(acc >= 0.98);
}
