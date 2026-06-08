cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0167 NEW) # Fix for INTERFACE_LINK_LIBRARIES generator expression.
cmake_policy(SET CMP0068 NEW) # Enables RPATH for macOS.

SET(PROJECT_NAME chinet)
PROJECT(${PROJECT_NAME})

# C++ standard
ENABLE_LANGUAGE(CXX)
SET(CMAKE_CXX_STANDARD 11)

# CMAKE
###########
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
MESSAGE(STATUS "CMAKE_MODULE_PATH='${CMAKE_MODULE_PATH}'")

################################################################################
# BUILD OPTIONS
################################################################################
OPTION(WITH_AVX "Compile tttrlib with AVX intrinsics" ON)
OPTION(INSTALL "Install the library and the include files" ON)
OPTION(BUILD_LIBRARY "Build the library and the include files" ON)
OPTION(BUILD_PYTHON_INTERFACE "Build Python interface" ON)
OPTION(BUILD_PYTHON_DOCS "Build documentation of SWIG Python interface" OFF)
OPTION(VERBOSE_CHINET "Verbose build" OFF)


## System specific compiler flags
################################
# AVX
IF((NOT CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64") OR APPLE)
    set(WITH_AVX OFF CACHE STRING "Disable AVX when compiling WITH_AVX=OFF." FORCE)
ENDIF()
IF(WITH_AVX)
    INCLUDE(simd)
    IF (MSVC)
        # https://devblogs.microsoft.com/cppblog/simd-extension-to-c-openmp-in-visual-studio/
        # /Oi is for intrinsics
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi")
    ELSE(MSVC)
        STRING(APPEND CMAKE_CXX_FLAGS " -O3 -mfma")
    ENDIF ()
ENDIF()

# Debug build
IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(VERBOSE_TTTRLIB ON)
    if(MSVC)
        ADD_COMPILE_OPTIONS(/Wall)
    ELSE(MSVC)
        ADD_COMPILE_OPTIONS(-Wall -Wextra -pedantic)
    ENDIF(MSVC)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug")

# Operation sytem specific flags
IF (MSVC)
    ADD_DEFINITIONS(-DMS_WIN64)
    ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
    # Issue with MSVC exceptions (https://stackoverflow.com/questions/6832666/lnk2019-when-including-asio-headers-solution-generated-with-cmake)
    # https://github.com/microsoft/vcpkg/discussions/25414
    ADD_DEFINITIONS(/DWIN32 /D_WINDOWS /GR /EHsc)
ENDIF (MSVC)
if(APPLE)
    FIND_PACKAGE(Threads)

    # comatibility with older systems
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.13")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.13")
    set(CMAKE_LFLAGS "${CMAKE_LFLAGS} -mmacosx-version-min=10.13")

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
    # explicit link to libc++ for recent osx versions
    MESSAGE(STATUS "Explicitly link to libc++ on modern osx versions")
    ADD_DEFINITIONS(-stdlib=libc++)
    SET(CMAKE_LFLAGS "${CMAKE_LFLAGS} -mmacosx-version-min=10.9")
    # Don't set MACOSX_RPATH by default
    # https://github.com/conda/conda/issues/3624
    if(DEFINED ENV{CONDA_PREFIX})
        set(MACOSX_RPATH OFF)
        set(CMAKE_SKIP_RPATH TRUE)
        set(CMAKE_SKIP_INSTALL_RPATH TRUE)
    endif()
endif()


#######################################
# INCLUDES AND LIBRARIES
#######################################

## Boost
###########################
IF(Boost_USE_STATIC_LIBS)
    SET(Boost_USE_STATIC_LIBS ON)
ENDIF(Boost_USE_STATIC_LIBS)
FIND_PACKAGE(Boost 1.36 REQUIRED COMPONENTS filesystem iostreams locale)
LINK_LIBRARIES(Boost::locale)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# RTTR
###########################
# Add the RTTR submodule directory
SET(RTTR_DIR ${CMAKE_SOURCE_DIR}/thirdparty/rttr)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rttr/src)

# Disable RTTR examples, tests, documentation, ...
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
set(CUSTOM_DOXYGEN_STYLE OFF CACHE BOOL "" FORCE)
set(BUILD_RTTR_DYNAMIC ON CACHE BOOL "" FORCE)
set(RTTR_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")

# Disable type conversion errors/warnings on osx
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-const-int-float-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=implicit-const-int-float-conversion")
endif()

add_subdirectory(${RTTR_DIR})
LINK_LIBRARIES(RTTR::Core)

# libmongoc
###########################
FIND_PACKAGE (libmongoc-1.0)
MESSAGE ("-- Found mongoc version \"${MONGOC_VERSION}\"")
MESSAGE ("-- include path \"${MONGOC_INCLUDE_DIRS}\"")
MESSAGE ("-- libraries \"${MONGOC_LIBRARIES}\"")
INCLUDE_DIRECTORIES (${MONGOC_INCLUDE_DIRS})
LINK_LIBRARIES (${MONGOC_LIBRARIES})
ADD_COMPILE_DEFINITIONS(${MONGOC_DEFINITIONS})

IF(VERBOSE_CHINET)
    ADD_COMPILE_DEFINITIONS(CHINET_VERBOSE)
ENDIF(VERBOSE_CHINET)

# Targets
#############################
FILE(GLOB_RECURSE SRC_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

ADD_SUBDIRECTORY(pyext)


# Print additional (Debug) info
#####################################
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    get_property(dirs DIRECTORY . PROPERTY INCLUDE_DIRECTORIES)
    foreach(dir ${dirs})
        message(STATUS "INCLUDE_DIRECTORY='${dir}'")
    endforeach()
    get_property(dirs DIRECTORY . PROPERTY LINK_LIBRARIES)
    foreach(dir ${dirs})
        message(STATUS "LINK_LIBRARIES='${dir}'")
    endforeach()
    get_property(dirs DIRECTORY . PROPERTY LINK_DIRECTORIES)
    foreach(dir ${dirs})
        message(STATUS "LINK_DIRECTORIES='${dir}'")
    endforeach()
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")


