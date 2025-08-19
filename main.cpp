#include "ml/include/LinearRegression.h"
#include "ml/include/LogisticRegression.h"
#include <iostream>

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
    LinearRegression::splitData(X, y, X_train, y_train, X_val, y_val, X_test, y_test);

    // Now you can use the split data
    LinearRegression lr;
    lr.train(X_train, y_train, 0.01, 1000);
    Eigen::VectorXd y_val_pred = lr.predict(X_val);
    Eigen::VectorXd y_test_pred = lr.predict(X_test);

    std::cout << "Validation predictions: \n" << y_val_pred << std::endl;
    std::cout << "Test predictions: \n" << y_test_pred << std::endl;
}

// Function to test the logistic-regression model.
void testLogisticRegression() {
    // Generate a simple linearly-separable dataset
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(200, 3);
    Eigen::VectorXd y = (X.rowwise().sum().array() > 0.0).cast<double>();

    Eigen::MatrixXd X_train, X_val, X_test;
    Eigen::VectorXd y_train, y_val, y_test;
    LogisticRegression::splitData(X, y, X_train, y_train, X_val, y_val, X_test, y_test);

    LogisticRegression clf;
    clf.train(X_train, y_train, 0.1, 2000);

    Eigen::VectorXd y_val_proba = clf.predict(X_val);
    Eigen::VectorXd y_test_proba = clf.predict(X_test);

    // Compute simple accuracies with 0.5 threshold
    auto compute_accuracy = [](const Eigen::VectorXd& proba, const Eigen::VectorXd& labels) {
        int correct = 0;
        for (int i = 0; i < proba.size(); ++i) {
            int pred = proba(i) >= 0.5 ? 1 : 0;
            if (pred == static_cast<int>(labels(i)))
                correct++;
        }
        return static_cast<double>(correct) / static_cast<double>(proba.size());
    };

    double val_acc = compute_accuracy(y_val_proba, y_val);
    double test_acc = compute_accuracy(y_test_proba, y_test);

    std::cout << "Logistic Regression Validation accuracy: " << val_acc << std::endl;
    std::cout << "Logistic Regression Test accuracy: " << test_acc << std::endl;
}

int main() {
    testLinearRegression();
    testLogisticRegression();
    return 0;
}
