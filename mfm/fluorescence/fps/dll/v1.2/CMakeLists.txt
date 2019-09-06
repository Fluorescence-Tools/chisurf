cmake_minimum_required(VERSION 3.6)
project(fpsdll)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES av_routines.cpp fpsnative.h fpsnative.h mtrandom.h msectim.h av_testbench.cpp av_routines.cpp mtrandom.h msectim.h)
add_library(fpsdll ${SOURCE_FILES})
add_executable(av_testbench ${SOURCE_FILES})