// Renderer.cpp

#include "Renderer.h"
#include <unordered_map>
#include <cstdio>      // for snprintf
#include <cmath>       // for std::fabs

#include "game/Snake.h" 
#include <raylib.h>
#include <algorithm>
#include <iostream>
using namespace render;
using namespace game;
using namespace neat;

Renderer::Renderer(int sw, int sh, int gw, int gh)
 : screenW_(sw), screenH_(sh), gridW_(gw), gridH_(gh)
{
    InitWindow(sw, sh, "Snake NEAT");
    SetTargetFPS(0);
}

Renderer::~Renderer(){
    CloseWindow();
}

void Renderer::beginFrame() { BeginDrawing(); ClearBackground(RAYWHITE); }
void Renderer::endFrame()   { EndDrawing(); }
bool Renderer::shouldClose(){ return WindowShouldClose(); }

void Renderer::drawGrid(){
    float cellW = float(screenW_/2)/gridW_, cellH = float(screenH_)/gridH_;
    
    for (int x = 0; x <= gridW_; ++x) {
        // cast ints to float so no narrowing in {}
        DrawLineV({ x * cellW, 0.0f },
                  { x * cellW, static_cast<float>(screenH_) },
                  LIGHTGRAY);
    }
    for (int y = 0; y <= gridH_; ++y) {
        DrawLineV({ 0.0f, y * cellH },
                  { static_cast<float>(screenW_/2), y * cellH },
                  LIGHTGRAY);
    }
}

void Renderer::drawSnake(const std::vector<game::Vec2i>& body) {
    float cellW = static_cast<float>(screenW_/2) / gridW_;
    float cellH = static_cast<float>(screenH_) / gridH_;
    for (auto& s : body) {
        DrawRectangle(
            static_cast<int>(s.x * cellW),
            static_cast<int>(s.y * cellH),
            static_cast<int>(cellW),
            static_cast<int>(cellH),
            GREEN
        );
    }
}


void Renderer::drawFood(int x, int y){
    float cellW = float(screenW_/2)/gridW_, cellH = float(screenH_)/gridH_;
    DrawRectangle(x*cellW, y*cellH, cellW, cellH, RED);
}

void Renderer::drawNetwork(const Network& net) {
    // Get the current genome (structure) and activations (neuron outputs)
    const auto& genome      = net.getGenome();
    const auto& activations = net.getActivations();

    // Separate all nodes into input, hidden, and output node ID lists
    std::vector<NodeId> inputs, hidden, outputs;
    for (const auto& kv : genome.nodes) {
        switch (kv.second.type) {
            case NodeGene::INPUT:  inputs.push_back(kv.first);  break;
            case NodeGene::BIAS:  inputs.push_back(kv.first);  break;
            case NodeGene::HIDDEN: hidden.push_back(kv.first);  break;
            case NodeGene::OUTPUT: outputs.push_back(kv.first); break;
        }
    }

    // Define the region on screen to draw the network (right 35% of window)
    float areaX = screenW_ * 0.55f;
    float areaY = screenH_ * 0.05f;
    float areaW = screenW_ * 0.4f;
    float areaH = screenH_ * 0.9f;

    // ---- Topological sort (layout columns) ----

    // 1. Initialize a "depth" map for all nodes
    //    Inputs always start at depth=0, hidden start at 1, outputs set later
    std::unordered_map<NodeId, int> depth;
    for (NodeId id : inputs)  depth[id] = 0;
    for (NodeId id : hidden)  depth[id] = 1;

    // 2. Propagate depths: for each enabled connection (not to outputs), 
    //    set the destination node's depth to at least one more than the source node
    int maxIters = static_cast<int>(genome.nodes.size());
    for (int iter = 0; iter < maxIters; ++iter) {
        bool anyChange = false;
        for (const auto& kv : genome.connections) {
            const auto& cg = kv.second;
            if (!cg.enabled) continue; // Only layout enabled connections

            // Ensure both source and target nodes exist
            auto fromIt = genome.nodes.find(cg.from);
            auto   toIt = genome.nodes.find(cg.to);
            if (fromIt == genome.nodes.end() || toIt == genome.nodes.end())
                continue; // skip if nodes are missing (shouldn't happen)

            // Only propagate depth to non-output nodes
            if (toIt->second.type == NodeGene::OUTPUT)
                continue;

            int d_from = depth[cg.from];      // Current depth of source node
            int& d_to  = depth[cg.to];        // Reference to depth of target node
            int want   = d_from + 1;          // The minimum depth the target should be
            if (d_to < want) {                // If we need to bump up its depth...
                d_to = want;                  // ...set it, and
                anyChange = true;             // ...note that something changed
            }
        }
        if (!anyChange) break; // Once no more changes, we're done
    }

    // 3. Find the maximum hidden depth reached (for # of columns)
    int maxHiddenDepth = 1;
    for (NodeId id : hidden)
        maxHiddenDepth = std::max(maxHiddenDepth, depth[id]);

    // 4. Calculate the number of layers/columns:
    //    (input layer) + (hidden layers) + (output layer)
    int totalLayers = maxHiddenDepth + 2; // +2 for input and output columns

    // 5. Bucket all nodes into their display columns ("layers")
    std::vector<std::vector<NodeId>> layers(totalLayers); // One vector per column
    for (NodeId id : inputs)
        layers[0].push_back(id); // Inputs in first column
    for (NodeId id : hidden)
        layers[depth[id]].push_back(id); // Hidden in their computed columns
    for (NodeId id : outputs)
        layers[totalLayers - 1].push_back(id); // Outputs in last column

    // 6. Assign screen coordinates for each node, spacing evenly vertically within its layer
    std::unordered_map<NodeId, Vector2> positions;
    for (int layerIdx = 0; layerIdx < totalLayers; ++layerIdx) {
        // Compute the x position for this layer/column
        float x = areaX + areaW * (float(layerIdx) / float(totalLayers - 1));
        auto& bucket = layers[layerIdx];
        std::sort(bucket.begin(), bucket.end()); // Sort for visual consistency

        size_t count = bucket.size(); // How many nodes in this layer
        for (size_t i = 0; i < count; ++i) {
            // Spread the nodes vertically, with equal spacing
            float y = areaY + areaH * (float(i + 1) / float(count + 1));
            positions[bucket[i]] = { x, y };
        }
    }

    // ---- Draw all enabled connections (edges) ----
    for (const auto& kv : genome.connections) {
        const auto& cg = kv.second;
        if (!cg.enabled) continue; // Only draw enabled connections

        // Find source and destination coordinates
        auto itA = positions.find(cg.from);
        auto itB = positions.find(cg.to);
        if (itA == positions.end() || itB == positions.end())
            continue; // Don't draw if one node is missing (shouldn't happen)

        // Draw the connection as a line (gray)
        DrawLineV(itA->second, itB->second, LIGHTGRAY);
    }

    // ---- Draw all nodes (neurons) ----
    const float nodeR = 5.0f; // Radius of each node's circle
    for (const auto& kv : positions) {
        NodeId id = kv.first;      // The node's ID
        Vector2 pos = kv.second;   // Its screen position

        // Look up activation value (or 0 if not found)
        float act = 0.0f;
        auto it = activations.find(id);
        if (it != activations.end())
            act = it->second;

        // Color: green for positive activation, red for negative, intensity is abs(activation)
        unsigned char inten = static_cast<unsigned char>(200 * std::fabs(act));
        Color col = (act >= 0.0f)
            ? Color{0, inten, 0, 255}
            : Color{inten, 0, 0, 255};

        // Draw filled circle
        DrawCircleV(pos, nodeR, col);

        // Draw outline
        DrawCircleLines(int(pos.x), int(pos.y), nodeR, BLACK);

        // Draw the activation value as text next to the node
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2f", act);
        DrawText(buf,
                 int(pos.x + nodeR + 2),
                 int(pos.y - 8),
                 10,
                 BLACK);
    }
}


void Renderer::drawStats(int gen, float maxF, float avgF, int speciesCount){
    DrawText(TextFormat("Gen: %d  MaxF: %.1f  AvgF: %.1f  Species: %d",
             gen, maxF, avgF, speciesCount), 10, 10, 20, BLACK);
}

void Renderer::processUI(bool& paused, float& speed, int& observeIdx){
    // e.g. keys to pause, speed up/down, change observed individual
    if (IsKeyPressed(KEY_SPACE)) paused = !paused;
    if (IsKeyPressed(KEY_UP)) speed = std::min(speed*2, 64.0f);
    if (IsKeyPressed(KEY_DOWN)) speed = std::max(speed/2, 0.25f);
    if (IsKeyPressed(KEY_RIGHT)) observeIdx++;
    if (IsKeyPressed(KEY_LEFT)) observeIdx--;
}