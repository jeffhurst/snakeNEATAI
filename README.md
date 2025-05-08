# Snake NEAT

A highly optimized C++17 implementation of the NEAT algorithm training an AI to play Snake, with Raylib visualization.

## Features

- Grid‐based Snake with food, self‐collision, and scoring  
- Full NEAT: speciation, crossover, mutation, fitness  
- Parallel evaluation via thread pool  
- Object pools for genomes & networks  
- Real‐time visualization: game + neural network  
- Export/load top genomes  
- UI: pause, speed control, observe individuals  

## Build

```bash
git clone "https://github.com/jeffhurst/snakeNEATAI"
cd snakeNEATAI
mkdir build && cd build
cmake ..
make -j
./SnakeNEAT
