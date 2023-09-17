//
// Created by Owen Carey on 8/13/23.
//

#ifndef CPP_AI_AUTOENCODER_H
#define CPP_AI_AUTOENCODER_H


class Autoencoder {
private:
    // TODO

public:
    // Default constructor
    Autoencoder();

    // Copy constructor
    Autoencoder(const Autoencoder& other);

    // Copy assignment operator
    Autoencoder& operator=(const Autoencoder& other);

    // Move constructor
    Autoencoder(Autoencoder&& other) noexcept;

    // Move assignment operator
    Autoencoder& operator=(Autoencoder&& other) noexcept;

    // Destructor
    ~Autoencoder();
};


#endif //CPP_AI_AUTOENCODER_H
