//
// Created by Owen Carey on 8/4/23.
//

#ifndef CPP_AI_SUPPORTVECTORMACHINE_H
#define CPP_AI_SUPPORTVECTORMACHINE_H

#include <Eigen/Dense>
#include <random>

class SupportVectorMachine {
  private:
    Eigen::VectorXd weights;
    double bias;

  public:
    // Default constructor
    SupportVectorMachine();

    // Copy constructor
    SupportVectorMachine(const SupportVectorMachine& other);

    // Copy assignment operator
    SupportVectorMachine& operator=(const SupportVectorMachine& other);

    // Move constructor
    SupportVectorMachine(SupportVectorMachine&& other) noexcept;

    // Move assignment operator
    SupportVectorMachine& operator=(SupportVectorMachine&& other) noexcept;

    // Destructor
    ~SupportVectorMachine();

    // Train linear SVM using hinge loss with L2 regularization (primal, batch subgradient)
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double learning_rate,
               int iterations, double C = 1.0);

    // Predict class labels in {0,1}
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;

    // Utility: split data into train/val/test with shuffling
    static void splitData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                          Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train,
                          Eigen::MatrixXd& X_val, Eigen::VectorXd& y_val, Eigen::MatrixXd& X_test,
                          Eigen::VectorXd& y_test, double train_size = 0.8,
                          double val_size = 0.1);
};

#endif // CPP_AI_SUPPORTVECTORMACHINE_H
