cmake_minimum_required(VERSION 3.21)
project(pq_reader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pq_reader STATIC
  parquet_f32.cpp
)

target_include_directories(pq_reader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

# Arrow/Parquet CMake targets följer mönstret Package::..._shared/_static
target_link_libraries(pq_reader PRIVATE
  Arrow::arrow_shared
  Parquet::parquet_shared
)
