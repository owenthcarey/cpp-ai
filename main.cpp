#include <iostream>
#include "ml/LinearRegression.h"

// Function to test the linear-regression model.
// This function generates a set of dummy data with 100 samples and 3 features each, where
// the target values are the sum of the features. It creates a LinearRegression model and
// trains it using the dummy data with a learning rate of 0.01 and 1000 iterations. Then
// it makes predictions on the same data and displays the predictions.
void testLinearRegression() {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 3);
    Eigen::VectorXd y = X.rowwise().sum();
    LinearRegression lr;
    lr.train(X, y, 0.01, 1000);
    Eigen::VectorXd y_pred = lr.predict(X);
    std::cout << "Predictions: \n" << y_pred << std::endl;
}

int main() {
    testLinearRegression();
    return 0;
}
