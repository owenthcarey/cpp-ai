//
// Created by Owen Carey on 8/12/23.
//

#ifndef CPP_AI_GENERATIVEADVERSARIALNETWORK_H
#define CPP_AI_GENERATIVEADVERSARIALNETWORK_H


class GenerativeAdversarialNetwork {
private:
    // TODO

public:
    // Default constructor
    GenerativeAdversarialNetwork();

    // Copy constructor
    GenerativeAdversarialNetwork(const GenerativeAdversarialNetwork &other);

    // Copy assignment operator
    GenerativeAdversarialNetwork &
    operator=(const GenerativeAdversarialNetwork &other);

    // Move constructor
    GenerativeAdversarialNetwork(GenerativeAdversarialNetwork &&other) noexcept;

    // Move assignment operator
    GenerativeAdversarialNetwork &
    operator=(GenerativeAdversarialNetwork &&other) noexcept;

    // Destructor
    ~GenerativeAdversarialNetwork();
};


#endif //CPP_AI_GENERATIVEADVERSARIALNETWORK_H
