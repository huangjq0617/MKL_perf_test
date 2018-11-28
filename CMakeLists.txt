cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunn C CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -fPIC -O3 -Ofast -m64 -march=native -funroll-loops -ffinite-math-only -Wno-unused-result -Wno-deprecated -pthread")

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14 -fPIC -g -O0 -m64  -march=native -funroll-loops -ffinite-math-only -Wno-unused-result -Wno-deprecated -pthread")
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE} -g -pg")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})

if(BUILD_STATIC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  set(CMAKE_EXE_LINKER_FLAGS "-static")
  set(Boost_USE_STATIC_LIBS  "TRUE")
endif(BUILD_STATIC)

include_directories(/opt/intel/mkl/include)
link_libraries("/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so" "/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so" "/opt/intel/mkl/lib/intel64/libmkl_core.so" iomp5 pthread m dl python2.7)

add_subdirectory(src)
