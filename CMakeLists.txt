cmake_minimum_required(VERSION 3.10)
project(SnakeNEAT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pipe")

find_package(raylib REQUIRED)

include_directories(src)

add_subdirectory(src/core)
add_subdirectory(src/game)
add_subdirectory(src/neat)
add_subdirectory(src/render)

add_executable(SnakeNEAT
    src/main.cpp
)
target_link_libraries(SnakeNEAT
    core
    game
    neat
    render
    raylib
    pthread
)
