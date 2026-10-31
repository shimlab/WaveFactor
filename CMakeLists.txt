cmake_minimum_required(VERSION 3.20)

project(WaveFactor LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags (portable -O3 on GCC/Clang, automatic AVX2/FMA detection with fallback, optional -march=native)
option(ENABLE_AVX2 "Enable AVX2 and FMA vectorization instructions when supported" ON)
option(ENABLE_NATIVE_TUNING "Enable host CPU architecture tuning (-march=native)" OFF)

set(WAVEFACTOR_COMPILE_OPTIONS "")
if(NOT MSVC)
    list(APPEND WAVEFACTOR_COMPILE_OPTIONS "$<$<NOT:$<CONFIG:Debug>>:-O3>")
endif()

if(ENABLE_AVX2)
    if(MSVC)
        set(_WAVEFACTOR_AVX2_REQ_FLAGS "/arch:AVX2")
        set(_WAVEFACTOR_AVX2_LIST_FLAGS /arch:AVX2)
    else()
        set(_WAVEFACTOR_AVX2_REQ_FLAGS "-mavx2 -mfma")
        set(_WAVEFACTOR_AVX2_LIST_FLAGS -mavx2 -mfma)
    endif()

    set(_WAVEFACTOR_AVX2_TEST_SRC "
        #include <immintrin.h>
        int main() {
            __m256d a = _mm256_set1_pd(1.0);
            __m256d b = _mm256_set1_pd(2.0);
            __m256d c = _mm256_fmadd_pd(a, b, a);
            double out[4];
            _mm256_storeu_pd(out, c);
            return (out[0] == 3.0) ? 0 : 1;
        }
    ")

    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "${_WAVEFACTOR_AVX2_REQ_FLAGS}")
    if(CMAKE_CROSSCOMPILING)
        include(CheckCXXSourceCompiles)
        check_cxx_source_compiles("${_WAVEFACTOR_AVX2_TEST_SRC}" WAVEFACTOR_HAS_AVX2)
    else()
        include(CheckCXXSourceRuns)
        check_cxx_source_runs("${_WAVEFACTOR_AVX2_TEST_SRC}" WAVEFACTOR_HAS_AVX2)
    endif()
    cmake_pop_check_state()

    if(WAVEFACTOR_HAS_AVX2)
        list(APPEND WAVEFACTOR_COMPILE_OPTIONS ${_WAVEFACTOR_AVX2_LIST_FLAGS})
        message(STATUS "WaveFactor: AVX2/FMA acceleration enabled (${_WAVEFACTOR_AVX2_REQ_FLAGS})")
    else()
        message(STATUS "WaveFactor: AVX2/FMA unavailable on target; falling back to baseline instructions")
    endif()
endif()

if(ENABLE_NATIVE_TUNING AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" WAVEFACTOR_HAS_MARCH_NATIVE)
    if(WAVEFACTOR_HAS_MARCH_NATIVE)
        list(APPEND WAVEFACTOR_COMPILE_OPTIONS -march=native)
    endif()
endif()

# Subdirectories for vendored libraries (isolated from user/system CMake registries)
option(BUILD_CPP_TESTS "Build C++ GoogleTest suite" ON)
set(CMAKE_EXPORT_PACKAGE_REGISTRY OFF)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_CMAKE_PACKAGE OFF CACHE BOOL "" FORCE)

# MSVC only exposes M_PI (used throughout src/) when _USE_MATH_DEFINES is defined
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

add_subdirectory(lib/eigen EXCLUDE_FROM_ALL)
add_subdirectory(src)

# Enable testing for C++ GoogleTest suite
if(BUILD_CPP_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest/CMakeLists.txt")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    # Use the shared MSVC runtime so GoogleTest links against the (/MD) WaveFactorSrc library
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(lib/googletest EXCLUDE_FROM_ALL)
    enable_testing()
    add_subdirectory(test)
endif()

# Headers and Sources for Pybind11 extension module
set(Headers
    src/cavi.hpp
    src/elbo.hpp
    src/init.hpp
    src/parameters.hpp
    src/tensor.hpp
    src/updates.hpp
    src/utilities.hpp
    src/bindings.hpp
)

set(Sources
    src/cavi.cpp
    src/elbo.cpp
    src/init.cpp
    src/parameters.cpp
    src/tensor.cpp
    src/updates.cpp
    src/utilities.cpp
    src/bindings.cpp
)

# Pybind11 module build configuration
option(BUILD_PYTHON_MODULE "Build Python Pybind11 extension module" ON)

if(BUILD_PYTHON_MODULE)
    set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use FindPython mode in pybind11" FORCE)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11/CMakeLists.txt")
        message(FATAL_ERROR "Vendored pybind11 not found at lib/pybind11/CMakeLists.txt")
    endif()
    add_subdirectory(lib/pybind11 EXCLUDE_FROM_ALL)

    pybind11_add_module(WaveFactor MODULE ${Sources} ${Headers})
    target_link_libraries(WaveFactor PRIVATE eigen)
    target_include_directories(WaveFactor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # Optimization flags
    target_compile_options(WaveFactor PRIVATE ${WAVEFACTOR_COMPILE_OPTIONS})

    # Static linking of GCC/MinGW runtimes on Windows to avoid external DLL dependencies
    if(MINGW)
        target_link_options(WaveFactor PRIVATE -static-libgcc -static-libstdc++)
    endif()

    # Create __init__.py in binary directory for testing direct imports from build
    set(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
    if(NOT EXISTS ${INIT_PY})
        file(WRITE ${INIT_PY} "")
    endif()
endif()
