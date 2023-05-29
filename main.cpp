#include <iostream>
#include "ml/LinearRegression.h"

int main() {
    // Generate dummy data
    // 100 samples, 3 features each
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 3);
    // Target values are sum of features
    Eigen::VectorXd y = X.rowwise().sum();
    // Create Linear Regression model
    LinearRegression lr;
    // Train the model with learning rate 0.01 and 1000 iterations
    lr.train(X, y, 0.01, 1000);
    // Make predictions on the same data
    Eigen::VectorXd y_pred = lr.predict(X);
    // Display the predictions
    std::cout << "Predictions: \n" << y_pred << std::endl;
    return 0;
}
