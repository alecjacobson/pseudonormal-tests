cmake_minimum_required(VERSION 3.18)
project(pseudonormal_precision LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
include(FetchContent)
FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.7.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nanobind)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0 GIT_SHALLOW TRUE)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

set(LIBIGL_SOURCE_DIR "" CACHE PATH "libigl checkout to measure")
set(BINDINGS_SOURCE_DIR "" CACHE PATH "libigl Python bindings checkout to measure")
set(PRECISION_MODULE "_precision" CACHE STRING "Import name of this build")
if(NOT EXISTS "${LIBIGL_SOURCE_DIR}/include/igl/signed_distance.h" OR
   NOT EXISTS "${BINDINGS_SOURCE_DIR}/src/signed_distance.cpp")
  message(FATAL_ERROR "Set LIBIGL_SOURCE_DIR and BINDINGS_SOURCE_DIR to source checkouts")
endif()
nanobind_add_module(${PRECISION_MODULE}
  module.cpp "${BINDINGS_SOURCE_DIR}/src/signed_distance.cpp")
target_compile_definitions(${PRECISION_MODULE} PRIVATE PRECISION_MODULE=${PRECISION_MODULE})
target_include_directories(${PRECISION_MODULE} PRIVATE
  "${LIBIGL_SOURCE_DIR}/include" "${BINDINGS_SOURCE_DIR}/include")
target_include_directories(${PRECISION_MODULE} SYSTEM PRIVATE "${eigen_SOURCE_DIR}")
# Keep compiler contraction consistent between the before/after builds.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(${PRECISION_MODULE} PRIVATE -ffp-contract=off)
endif()

option(PRECISION_BUILD_TESTS "Build the affected upstream C++ tests" OFF)
if(PRECISION_BUILD_TESTS)
  FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v2.13.8 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(catch2)
  add_executable(test_precision
    "${LIBIGL_SOURCE_DIR}/tests/main.cpp"
    "${LIBIGL_SOURCE_DIR}/tests/include/igl/signed_distance.cpp"
    "${LIBIGL_SOURCE_DIR}/tests/include/igl/pseudonormal_test.cpp"
    "${LIBIGL_SOURCE_DIR}/tests/include/igl/point_simplex_squared_distance.cpp")
  target_include_directories(test_precision PRIVATE
    "${LIBIGL_SOURCE_DIR}/include" "${LIBIGL_SOURCE_DIR}/tests")
  target_include_directories(test_precision SYSTEM PRIVATE "${eigen_SOURCE_DIR}")
  target_compile_definitions(test_precision PRIVATE LIBIGL_DATA_DIR="${LIBIGL_SOURCE_DIR}/tests/data")
  target_link_libraries(test_precision PRIVATE Catch2::Catch2)
endif()

option(PRECISION_BUILD_TOLERANCE_SWEEP "Build the runtime tolerance experiment" OFF)
if(PRECISION_BUILD_TOLERANCE_SWEEP)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/make_runtime_classifier.py"
      "${LIBIGL_SOURCE_DIR}/include/igl/pseudonormal_test.cpp"
      "${CMAKE_CURRENT_BINARY_DIR}/pseudonormal_runtime.h"
    RESULT_VARIABLE classifier_generation_result)
  if(NOT classifier_generation_result EQUAL 0)
    message(FATAL_ERROR "Could not generate the runtime pseudonormal classifier")
  endif()
  nanobind_add_module(_tolerance_sweep tolerance_sweep.cpp)
  target_include_directories(_tolerance_sweep PRIVATE
    "${LIBIGL_SOURCE_DIR}/include" "${BINDINGS_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}")
  target_include_directories(_tolerance_sweep SYSTEM PRIVATE "${eigen_SOURCE_DIR}")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_tolerance_sweep PRIVATE -ffp-contract=off)
  endif()
endif()
