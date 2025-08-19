//
// Created by Owen Carey on 8/9/23.
//

#ifndef CPP_AI_NEURALNETWORK_H
#define CPP_AI_NEURALNETWORK_H

class NeuralNetwork {
  private:
    // TODO

  public:
    // Default constructor
    NeuralNetwork();

    // Copy constructor
    NeuralNetwork(const NeuralNetwork& other);

    // Copy assignment operator
    NeuralNetwork& operator=(const NeuralNetwork& other);

    // Move constructor
    NeuralNetwork(NeuralNetwork&& other) noexcept;

    // Move assignment operator
    NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;

    // Destructor
    ~NeuralNetwork();
};

#endif // CPP_AI_NEURALNETWORK_H
