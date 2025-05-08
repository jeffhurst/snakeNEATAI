#pragma once
#include <vector>
#include "game/Snake.h"
#include "neat/Network.h"

namespace render {

class Renderer {
public:
    Renderer(int screenW, int screenH, int gridW, int gridH);
    ~Renderer();
    void beginFrame();
    void drawGrid();
    void drawSnake(const std::vector<game::Vec2i>& body);
    void drawFood(int x, int y);
    void drawNetwork(const neat::Network& net);
    void drawStats(int gen, float maxF, float avgF, int speciesCount);
    void endFrame();
    bool shouldClose();
    void processUI(bool& paused, float& speed, int& observeIdx);
private:
    int screenW_, screenH_, gridW_, gridH_;
};

} // namespace render
