cmake_minimum_required(VERSION 3.25)

# The path to vcpkg.cmake file provided by `vcpkg integrate install`
set(CMAKE_TOOLCHAIN_FILE /home/owenthcarey/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(cpp_ai)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 CONFIG REQUIRED)

add_executable(cpp_ai main.cpp ml/LinearRegression.cpp ml/LinearRegression.h)

# Link Eigen3 to your target
target_link_libraries(cpp_ai Eigen3::Eigen)
