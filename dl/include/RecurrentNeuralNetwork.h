//
// Created by Owen Carey on 8/11/23.
//

#ifndef CPP_AI_RECURRENTNEURALNETWORK_H
#define CPP_AI_RECURRENTNEURALNETWORK_H


class RecurrentNeuralNetwork {
private:
    // TODO

public:
    // Default constructor
    RecurrentNeuralNetwork();

    // Copy constructor
    RecurrentNeuralNetwork(const RecurrentNeuralNetwork& other);

    // Copy assignment operator
    RecurrentNeuralNetwork& operator=(const RecurrentNeuralNetwork& other);

    // Move constructor
    RecurrentNeuralNetwork(RecurrentNeuralNetwork&& other) noexcept;

    // Move assignment operator
    RecurrentNeuralNetwork& operator=(RecurrentNeuralNetwork&& other) noexcept;

    // Destructor
    ~RecurrentNeuralNetwork();
};


#endif //CPP_AI_RECURRENTNEURALNETWORK_H
