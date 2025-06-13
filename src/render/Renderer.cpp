// Renderer.cpp

#include "Renderer.h"
#include <unordered_map>
#include <cstdio>      // for snprintf
#include <cmath>       // for std::fabs

#include "game/Snake.h" 
#include <raylib.h>
#include <algorithm>
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

void Renderer::drawNetwork(const Network& net){
    // Retrieve structure and activations
    const auto& genome      = net.getGenome();
    const auto& activations = net.getActivations();

    // Separate nodes by layer
    std::vector<NodeId> inputs, hidden, outputs;
    for (auto& kv : genome.nodes) {
        switch (kv.second.type) {
            case NodeGene::INPUT:  inputs.push_back(kv.first);   break;
            case NodeGene::HIDDEN: hidden.push_back(kv.first);   break;
            case NodeGene::OUTPUT: outputs.push_back(kv.first);  break;
        }
    }
    // Sort IDs for consistent layout
    auto sortIds = [](auto& v){ std::sort(v.begin(), v.end()); };
    sortIds(inputs); sortIds(hidden); sortIds(outputs);

    // Define drawing area: right 35% of screen
    float areaX = screenW_ * 0.65f;
    float areaY = screenH_ * 0.1f;
    float areaW = screenW_ * 0.3f;
    float areaH = screenH_ * 0.8f;

    // X positions for each layer
    float layerX[3] = { areaX,
                        areaX + areaW * 0.5f,
                        areaX + areaW };

    // Map each node to a screen coordinate
    std::unordered_map<NodeId, Vector2> positions;
    auto layoutLayer = [&](const std::vector<NodeId>& layer, int idx) {
        size_t count = layer.size();
        for (size_t i = 0; i < count; ++i) {
            float x = layerX[idx];
            float y = areaY + areaH * (i + 1) / float(count + 1);
            positions[layer[i]] = { x, y };
        }
    };
    layoutLayer(inputs, 0);
    layoutLayer(hidden, 1);
    layoutLayer(outputs, 2);

    // Draw all enabled connections
    for (auto& kv : genome.connections) {
        const auto& cg = kv.second;
        if (!cg.enabled) continue;
        auto itA = positions.find(cg.from);
        auto itB = positions.find(cg.to);
        if (itA == positions.end() || itB == positions.end()) continue;
        DrawLineV(itA->second, itB->second, LIGHTGRAY);
    }

    // Draw nodes with activation-based coloring and value labels
    const float nodeR = 5.0f;
    for (auto& kv : positions) {
        NodeId id = kv.first;
        Vector2 pos = kv.second;
        float act = 0.0f;
        auto it = activations.find(id);
        if (it != activations.end()) act = it->second;

        // Color: green for positive, red for negative, intensity ~ magnitude
        unsigned char intensity = static_cast<unsigned char>(200 * std::fabs(act));
        Color col = (act >= 0.0f)
            ? Color{ 0, intensity, 0, 255 }
            : Color{ intensity, 0, 0, 255 };
        DrawCircleV(pos, nodeR, col);
        DrawCircleLines((int)pos.x, (int)pos.y, nodeR, BLACK);

        // Draw activation value text
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2f", act);
        DrawText(buf,
                 (int)(pos.x + nodeR + 2),
                 (int)(pos.y - 8),
                 10,
                 BLACK);
        // DrawText(buf,
        //          int(pos.x + nodeR + 2),
        //          int(pos.y - (nodeR + 2)),
        //          14,
        //          BLACK);
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
