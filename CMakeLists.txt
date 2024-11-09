cmake_minimum_required(VERSION 3.16)
project(number_classificator)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

find_package(SFML 2.5 COMPONENTS system window graphics network audio REQUIRED)
set(EIGEN3_INCLUDE_DIR "eigen-3.3.9")

add_executable(ProjExec main.cpp regression/mnist_regression.cpp)
target_include_directories(ProjExec PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(ProjExec sfml-system sfml-window sfml-graphics sfml-network sfml-audio)