cmake_minimum_required(VERSION 3.25)

# The path to vcpkg.cmake file provided by `vcpkg integrate install`
set(CMAKE_TOOLCHAIN_FILE /Users/owenthcarey/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(cpp_ai)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 CONFIG REQUIRED)

add_executable(cpp_ai main.cpp ml/src/LinearRegression.cpp ml/include/LinearRegression.h
        ml/src/LogisticRegression.cpp
        ml/include/LogisticRegression.h
        ml/src/KNearestNeighbors.cpp
        ml/include/KNearestNeighbors.h
        ml/src/SupportVectorMachine.cpp
        ml/include/SupportVectorMachine.h
        ml/src/KMeansClustering.cpp
        ml/include/KMeansClustering.h
        dl/src/NeuralNetwork.cpp
        dl/include/NeuralNetwork.h
        dl/src/ConvolutionalNeuralNetwork.cpp
        dl/include/ConvolutionalNeuralNetwork.h
        dl/src/RecurrentNeuralNetwork.cpp
        dl/include/RecurrentNeuralNetwork.h
        dl/src/GenerativeAdversarialNetwork.cpp
        dl/include/GenerativeAdversarialNetwork.h
        dl/src/Autoencoder.cpp
        dl/include/Autoencoder.h)

# Link Eigen3 to your target
target_link_libraries(cpp_ai Eigen3::Eigen)
