cmake_minimum_required(VERSION 3.21)
project(pq_reader LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
endif()

include(FetchContent)

option(PQ_READER_FETCH_ARROW "Fetch Apache Arrow/Parquet if not found" ON)

find_package(Arrow CONFIG QUIET)
find_package(Parquet CONFIG QUIET)

if(NOT Arrow_FOUND OR NOT Parquet_FOUND)
  if(NOT PQ_READER_FETCH_ARROW)
    message(FATAL_ERROR "Arrow/Parquet not found. Install them or enable PQ_READER_FETCH_ARROW.")
  endif()

  message(STATUS "Arrow/Parquet not found. Fetching Apache Arrow...")
  set(ARROW_BUILD_SHARED ON CACHE BOOL "" FORCE)
  set(ARROW_BUILD_STATIC OFF CACHE BOOL "" FORCE)
  # IPC is required for Parquet reader/writer functionality.
  set(ARROW_IPC ON CACHE BOOL "" FORCE)
  set(ARROW_PARQUET ON CACHE BOOL "" FORCE)
  set(ARROW_SIMD_LEVEL "NONE" CACHE STRING "" FORCE)
  set(ARROW_WITH_BZ2 OFF CACHE BOOL "" FORCE)
  set(ARROW_WITH_LZ4 OFF CACHE BOOL "" FORCE)
  # Pandas/pyarrow skriver ofta Parquet med snappy som default.
  set(ARROW_WITH_SNAPPY ON CACHE BOOL "" FORCE)
  # Zstd är också vanligt och bra att ha.
  set(ARROW_WITH_ZSTD ON CACHE BOOL "" FORCE)
  set(ARROW_WITH_BROTLI OFF CACHE BOOL "" FORCE)
  set(ARROW_DEPENDENCY_SOURCE "AUTO" CACHE STRING "" FORCE)

  FetchContent_Declare(
    arrow
    GIT_REPOSITORY https://github.com/apache/arrow.git
    GIT_TAG apache-arrow-15.0.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cpp
  )
  FetchContent_MakeAvailable(arrow)
endif()

# Some Arrow builds export different target names; these aliases keep the rest stable.
if(TARGET arrow_shared AND NOT TARGET Arrow::arrow_shared)
  add_library(Arrow::arrow_shared ALIAS arrow_shared)
endif()
if(TARGET arrow_static AND NOT TARGET Arrow::arrow_static)
  add_library(Arrow::arrow_static ALIAS arrow_static)
endif()
if(TARGET parquet_shared AND NOT TARGET Parquet::parquet_shared)
  add_library(Parquet::parquet_shared ALIAS parquet_shared)
endif()
if(TARGET parquet_static AND NOT TARGET Parquet::parquet_static)
  add_library(Parquet::parquet_static ALIAS parquet_static)
endif()

if(TARGET Arrow::arrow_shared)
  set(PQ_ARROW_TARGET Arrow::arrow_shared)
elseif(TARGET Arrow::arrow_static)
  set(PQ_ARROW_TARGET Arrow::arrow_static)
else()
  message(FATAL_ERROR "Arrow targets not found after configuration.")
endif()

if(TARGET Parquet::parquet_shared)
  set(PQ_PARQUET_TARGET Parquet::parquet_shared)
elseif(TARGET Parquet::parquet_static)
  set(PQ_PARQUET_TARGET Parquet::parquet_static)
else()
  message(FATAL_ERROR "Parquet targets not found after configuration.")
endif()

set(PQ_ARROW_INCLUDE_DIRS "")
if(DEFINED arrow_SOURCE_DIR)
  list(APPEND PQ_ARROW_INCLUDE_DIRS
    "${arrow_SOURCE_DIR}/cpp/src"
    "${arrow_BINARY_DIR}/src"
  )
endif()

add_library(pq_reader SHARED
  parquet_f32.cpp
)

target_include_directories(pq_reader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PQ_ARROW_INCLUDE_DIRS}
)

# Arrow/Parquet CMake targets följer mönstret Package::..._shared/_static
target_link_libraries(pq_reader PRIVATE
  ${PQ_ARROW_TARGET}
  ${PQ_PARQUET_TARGET}
)

enable_testing()

add_executable(pq_reader_test
  tests/test_parquet_reader.cpp
)

target_link_libraries(pq_reader_test PRIVATE
  pq_reader
  ${PQ_ARROW_TARGET}
  ${PQ_PARQUET_TARGET}
)

target_include_directories(pq_reader_test PRIVATE
  ${PQ_ARROW_INCLUDE_DIRS}
)

add_test(NAME pq_reader_test COMMAND pq_reader_test)
