//
// Created by owenthcarey on 5/28/23.
//

#ifndef CPP_AI_LINEARREGRESSION_H
#define CPP_AI_LINEARREGRESSION_H

#include <Eigen/Dense>
#include <random>

class LinearRegression {
private:
    Eigen::VectorXd weights;
    double bias;

public:
    LinearRegression();
    void train(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
               double learning_rate, int iterations);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    static void splitData(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                          Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train,
                          Eigen::MatrixXd &X_val, Eigen::VectorXd &y_val,
                          Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test,
                          double train_size = 0.8, double val_size = 0.1);
};

#endif //CPP_AI_LINEARREGRESSION_H
