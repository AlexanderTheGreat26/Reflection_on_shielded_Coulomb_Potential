cmake_minimum_required(VERSION 3.10.2)
project(Reflection_on_shielded_Coulomb_Potential)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -static-libstdc++ -static-libgcc")


add_executable(Reflection_on_shielded_Coulomb_Potential main.cpp)

set(SOURCE_FILES main.cpp)

add_executable(MyTarget ${SOURCE_FILES})