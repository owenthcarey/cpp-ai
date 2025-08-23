#include <Eigen/Dense>
#include <SupportVectorMachine.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("SupportVectorMachine classifies linearly separable data", "[svm]") {
    // Create a simple 2D dataset: class 1 if x0 - x1 >= margin, else class 0
    const int n = 300;
    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        double a = static_cast<double>(i % 60) / 60.0;       // 0..1 cyclic
        double b = static_cast<double>((i * 11) % 60) / 60.0; // pseudo-random-ish
        X(i, 0) = a;
        X(i, 1) = b;
        y(i) = (a - b >= 0.05) ? 1.0 : 0.0;
    }

    Eigen::MatrixXd X_train, X_val, X_test;
    Eigen::VectorXd y_train, y_val, y_test;
    SupportVectorMachine::splitData(X, y, X_train, y_train, X_val, y_val, X_test, y_test);

    SupportVectorMachine svm;
    svm.train(X_train, y_train, /*learning_rate=*/0.5, /*iterations=*/1500, /*C=*/1.0);

    auto accuracy = [](const Eigen::VectorXd& preds, const Eigen::VectorXd& labels) {
        return (preds.array() == labels.array()).cast<double>().mean();
    };

    Eigen::VectorXd val_preds = svm.predict(X_val);
    Eigen::VectorXd test_preds = svm.predict(X_test);
    REQUIRE(accuracy(val_preds, y_val) >= 0.97);
    REQUIRE(accuracy(test_preds, y_test) >= 0.97);
}
