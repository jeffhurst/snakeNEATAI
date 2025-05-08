#include "Renderer.h"
#include <raylib.h>
#include <algorithm>
using namespace render;
using namespace game;
using namespace neat;

Renderer::Renderer(int sw, int sh, int gw, int gh)
 : screenW_(sw), screenH_(sh), gridW_(gw), gridH_(gh)
{
    InitWindow(sw, sh, "Snake NEAT");
    SetTargetFPS(60);
}

Renderer::~Renderer(){
    CloseWindow();
}

void Renderer::beginFrame() { BeginDrawing(); ClearBackground(RAYWHITE); }
void Renderer::endFrame()   { EndDrawing(); }
bool Renderer::shouldClose(){ return WindowShouldClose(); }

void Renderer::drawGrid(){
    float cellW = float(screenW_)/gridW_, cellH = float(screenH_)/gridH_;
    for (int x=0;x<=gridW_;++x)
        DrawLineV({x*cellW,0},{x*cellW,screenH_}, LIGHTGRAY);
    for (int y=0;y<=gridH_;++y)
        DrawLineV({0,y*cellH},{screenW_,y*cellH}, LIGHTGRAY);
}

void Renderer::drawSnake(const std::vector<Snake::Vec2i>& body){
    float cellW = float(screenW_)/gridW_, cellH = float(screenH_)/gridH_;
    for (auto& s : body) {
        DrawRectangle(s.x*cellW, s.y*cellH, cellW, cellH, GREEN);
    }
}

void Renderer::drawFood(int x, int y){
    float cellW = float(screenW_)/gridW_, cellH = float(screenH_)/gridH_;
    DrawRectangle(x*cellW, y*cellH, cellW, cellH, RED);
}

void Renderer::drawNetwork(const Network& net){
    // very simple layout: layers vertically
    // Omitted for brevity; assume implemented
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
