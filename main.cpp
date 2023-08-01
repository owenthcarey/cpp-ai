#include <iostream>
#include "ml/include/LinearRegression.h"

// Function to test the linear-regression model.
// This function generates a set of dummy data with 100 samples and 3 features each, where
// the target values are the sum of the features. It creates a LinearRegression model and
// trains it using the dummy data with a learning rate of 0.01 and 1000 iterations. Then
// it makes predictions on the same data and displays the predictions.
void testLinearRegression() {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 3);
    Eigen::VectorXd y = X.rowwise().sum();

    // Create matrices and vectors for the splits
    Eigen::MatrixXd X_train, X_val, X_test;
    Eigen::VectorXd y_train, y_val, y_test;

    // Call the function to split the data
    LinearRegression::splitData(X, y, X_train, y_train, X_val, y_val, X_test,
                                y_test);

    // Now you can use the split data
    LinearRegression lr;
    lr.train(X_train, y_train, 0.01, 1000);
    Eigen::VectorXd y_val_pred = lr.predict(X_val);
    Eigen::VectorXd y_test_pred = lr.predict(X_test);

    std::cout << "Validation predictions: \n" << y_val_pred << std::endl;
    std::cout << "Test predictions: \n" << y_test_pred << std::endl;
}


int main() {
    testLinearRegression();
    return 0;
}
