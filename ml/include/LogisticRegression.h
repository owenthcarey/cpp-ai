//
// Created by Owen Carey on 8/2/23.
//

#ifndef CPP_AI_LOGISTICREGRESSION_H
#define CPP_AI_LOGISTICREGRESSION_H


class LogisticRegression {
private:
    // TODO

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
};


#endif //CPP_AI_LOGISTICREGRESSION_H
