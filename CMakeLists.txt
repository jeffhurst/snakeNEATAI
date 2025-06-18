cmake_minimum_required(VERSION 3.10)
project(SnakeNEAT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pipe")
endif()

# where you installed raylib
if (NOT raylib_DIR)
  set(raylib_DIR "C:/raylib/install/lib/cmake/raylib")
endif()
find_package(raylib REQUIRED)

# gather sources
set(GAME_SRCS
    src/game/Snake.cpp
    src/game/Game.cpp
)
set(NEAT_SRCS
    src/neat/Genome.cpp
    src/neat/Network.cpp
    src/neat/InnovationTracker.cpp
    src/neat/NEAT.cpp
    src/neat/Species.cpp
)
set(RENDER_SRCS
    src/render/Renderer.cpp
)
set(MAIN_SRCS
    src/main.cpp
)

add_executable(SnakeNEAT
    ${GAME_SRCS}
    ${NEAT_SRCS}
    ${RENDER_SRCS}
    src/main.cpp
)

# include directories
target_include_directories(SnakeNEAT PRIVATE
    src
    src/game
    src/neat
    src/render
)

find_package(raylib REQUIRED)


target_include_directories(SnakeNEAT PRIVATE src)

target_link_libraries(SnakeNEAT PRIVATE
    raylib
)

# On Windows, pull in Winmm for timing
if (WIN32)
  target_link_libraries(SnakeNEAT PRIVATE Winmm)
else()
  # On Linux/macOS, link the pthreads library properly
  find_package(Threads REQUIRED)
  target_link_libraries(SnakeNEAT PRIVATE Threads::Threads)
endif()
