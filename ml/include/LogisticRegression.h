//
// Created by Owen Carey on 8/2/23.
//

#ifndef CPP_AI_LOGISTICREGRESSION_H
#define CPP_AI_LOGISTICREGRESSION_H

#include <Eigen/Dense>
#include <random>

class LogisticRegression {
  private:
    Eigen::VectorXd weights;
    double bias;

  public:
    // Default constructor
    LogisticRegression();

    // Copy constructor
    LogisticRegression(const LogisticRegression& other);

    // Copy assignment operator
    LogisticRegression& operator=(const LogisticRegression& other);

    // Move constructor
    LogisticRegression(LogisticRegression&& other) noexcept;

    // Move assignment operator
    LogisticRegression& operator=(LogisticRegression&& other) noexcept;

    // Destructor
    ~LogisticRegression();

    void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double learning_rate,
               int iterations);

    // Returns probabilities P(y=1|x)
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;

    static void splitData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                          Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train,
                          Eigen::MatrixXd& X_val, Eigen::VectorXd& y_val, Eigen::MatrixXd& X_test,
                          Eigen::VectorXd& y_test, double train_size = 0.8, double val_size = 0.1);
};

#endif // CPP_AI_LOGISTICREGRESSION_H
