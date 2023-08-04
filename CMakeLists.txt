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
        ml/include/KNearestNeighbors.h)

# Link Eigen3 to your target
target_link_libraries(cpp_ai Eigen3::Eigen)
