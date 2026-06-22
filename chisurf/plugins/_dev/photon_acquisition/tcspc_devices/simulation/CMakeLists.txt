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
)

add_library(burbulator SHARED ${BURBULATOR_SOURCES})

target_include_directories(burbulator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# On Windows, also use the local .def file to export the expected symbols
if (WIN32)
    set(BURBULATOR_DEF "${CMAKE_CURRENT_SOURCE_DIR}/burbulator.def")
    if (EXISTS "${BURBULATOR_DEF}")
        target_sources(burbulator PRIVATE "${BURBULATOR_DEF}")
    else()
        message(WARNING "burbulator.def not found in ${CMAKE_CURRENT_SOURCE_DIR}; exports will use default visibility")
    endif()
endif()

# Place the built library directly into this simulation directory so Python can load it.
set_target_properties(burbulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
