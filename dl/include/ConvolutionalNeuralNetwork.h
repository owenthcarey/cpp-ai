//
// Created by Owen Carey on 8/10/23.
//

#ifndef CPP_AI_CONVOLUTIONALNEURALNETWORK_H
#define CPP_AI_CONVOLUTIONALNEURALNETWORK_H


class ConvolutionalNeuralNetwork {
private:
    // TODO

public:
    // Default constructor
    ConvolutionalNeuralNetwork();

    // Copy constructor
    ConvolutionalNeuralNetwork(const ConvolutionalNeuralNetwork& other);

    // Copy assignment operator
    ConvolutionalNeuralNetwork& operator=(const ConvolutionalNeuralNetwork& other);

    // Move constructor
    ConvolutionalNeuralNetwork(ConvolutionalNeuralNetwork&& other) noexcept;

    // Move assignment operator
    ConvolutionalNeuralNetwork& operator=(ConvolutionalNeuralNetwork&& other) noexcept;

    // Destructor
    ~ConvolutionalNeuralNetwork();
};


#endif //CPP_AI_CONVOLUTIONALNEURALNETWORK_H
