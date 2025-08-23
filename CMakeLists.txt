cmake_minimum_required(VERSION 3.25)

# Prefer toolchain provided by environment (e.g., CI). Fallback to prior local path if present.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    elseif(EXISTS "/Users/owenthcarey/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "/Users/owenthcarey/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    endif()
endif()

project(cpp_ai)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 CONFIG REQUIRED)

# Optionally include WIP deep-learning sources that currently contain placeholders
option(CPP_AI_BUILD_DL "Build deep learning sources (WIP)" OFF)

set(ML_LIB_SOURCES
        ml/src/LinearRegression.cpp
        ml/include/LinearRegression.h
        ml/src/LogisticRegression.cpp
        ml/include/LogisticRegression.h
        ml/src/KNearestNeighbors.cpp
        ml/include/KNearestNeighbors.h
        ml/src/SupportVectorMachine.cpp
        ml/include/SupportVectorMachine.h
        ml/src/KMeansClustering.cpp
        ml/include/KMeansClustering.h)

set(DL_SOURCES
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

add_library(cpp_ai_lib ${ML_LIB_SOURCES})
target_include_directories(cpp_ai_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ml/include)
target_link_libraries(cpp_ai_lib PUBLIC Eigen3::Eigen)

add_executable(cpp_ai main.cpp)
target_link_libraries(cpp_ai PRIVATE cpp_ai_lib)

if(CPP_AI_BUILD_DL)
    target_sources(cpp_ai_lib PRIVATE ${DL_SOURCES})
endif()

# Testing
include(CTest)
if(BUILD_TESTING)
    find_package(Catch2 3 CONFIG REQUIRED)
    add_executable(cpp_ai_tests
            tests/test_linear_regression.cpp
            tests/test_logistic_regression.cpp
            tests/test_knn.cpp
            tests/test_kmeans.cpp)
    target_link_libraries(cpp_ai_tests PRIVATE cpp_ai_lib Catch2::Catch2WithMain Eigen3::Eigen)
    include(Catch)
    catch_discover_tests(cpp_ai_tests)
endif()
