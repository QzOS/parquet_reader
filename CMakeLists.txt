cmake_minimum_required(VERSION 3.21)
project(pq_reader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

add_library(pq_reader SHARED
  parquet_f32.cpp
)

set_target_properties(pq_reader PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(pq_reader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Arrow/Parquet CMake targets följer mönstret Package::..._shared/_static
target_link_libraries(pq_reader PRIVATE
  Arrow::arrow_shared
  Parquet::parquet_shared
)

enable_testing()

add_executable(pq_reader_test
  tests/test_parquet_reader.cpp
)

target_link_libraries(pq_reader_test PRIVATE
  pq_reader
  Arrow::arrow_shared
  Parquet::parquet_shared
)

add_test(NAME pq_reader_test COMMAND pq_reader_test)
