cmake_minimum_required(VERSION 3.12)
project(burbulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BURBULATOR_SOURCES
    mt19937cok.cpp
    smdif_misc.cpp
    focus.cpp
    rotdiff.cpp
    smdif_ov3.cpp
    data2spc_tac.cpp
    burbulator_exports.cpp
)

add_library(burbulator SHARED ${BURBULATOR_SOURCES})

target_include_directories(burbulator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    set(BURBULATOR_DEF "${CMAKE_CURRENT_SOURCE_DIR}/burbulator.def")
    if(EXISTS "${BURBULATOR_DEF}")
        target_sources(burbulator PRIVATE "${BURBULATOR_DEF}")
    endif()
endif()

# Output directly to the build-tree bin dir so the copy step in setup.py
# can find it at a predictable location.
set_target_properties(burbulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
