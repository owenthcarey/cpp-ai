cmake_minimum_required(VERSION 3.25)
project(cpp_ai)

set(CMAKE_CXX_STANDARD 23)

include_directories("/usr/include/eigen3")

add_executable(cpp_ai main.cpp ml/LinearRegression.cpp ml/LinearRegression.h)
