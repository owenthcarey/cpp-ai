//
// Created by Owen Carey on 8/5/23.
//

#ifndef CPP_AI_KMEANSCLUSTERING_H
#define CPP_AI_KMEANSCLUSTERING_H


class KMeansClustering {
private:
    // TODO

public:
    // Default constructor
    KMeansClustering();

    // Copy constructor
    KMeansClustering(const KMeansClustering& other);

    // Copy assignment operator
    KMeansClustering& operator=(const KMeansClustering& other);

    // Move constructor
    KMeansClustering(KMeansClustering&& other) noexcept;

    // Move assignment operator
    KMeansClustering& operator=(KMeansClustering&& other) noexcept;

    // Destructor
    ~KMeansClustering();
};


#endif //CPP_AI_KMEANSCLUSTERING_H
