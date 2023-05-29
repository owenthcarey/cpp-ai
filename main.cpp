#include <iostream>
#include "ml/LinearRegression.h"

int main() {
    // Generate dummy data
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 3); // 100 samples, 3 features each
    Eigen::VectorXd y = X.rowwise().sum(); // Target values are sum of features

    LinearRegression lr;
    lr.train(X, y, 0.01, 1000);  // Train model with learning rate 0.01 and 1000 iterations

    Eigen::VectorXd y_pred = lr.predict(X);  // Make predictions on the same data

    std::cout << "Predictions: \n" << y_pred << std::endl;
    return 0;
}
