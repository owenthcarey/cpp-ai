//
// Created by Owen Carey on 8/4/23.
//

#ifndef CPP_AI_SUPPORTVECTORMACHINE_H
#define CPP_AI_SUPPORTVECTORMACHINE_H

class SupportVectorMachine {
  private:
    // TODO

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
};

#endif // CPP_AI_SUPPORTVECTORMACHINE_H
