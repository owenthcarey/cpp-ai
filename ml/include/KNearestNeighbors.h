//
// Created by Owen Carey on 8/3/23.
//

#ifndef CPP_AI_KNEARESTNEIGHBORS_H
#define CPP_AI_KNEARESTNEIGHBORS_H

class KNearestNeighbors {
  private:
    // TODO

  public:
    // Default constructor
    KNearestNeighbors();

    // Copy constructor
    KNearestNeighbors(const KNearestNeighbors& other);

    // Copy assignment operator
    KNearestNeighbors& operator=(const KNearestNeighbors& other);

    // Move constructor
    KNearestNeighbors(KNearestNeighbors&& other) noexcept;

    // Move assignment operator
    KNearestNeighbors& operator=(KNearestNeighbors&& other) noexcept;

    // Destructor
    ~KNearestNeighbors();
};

#endif // CPP_AI_KNEARESTNEIGHBORS_H
