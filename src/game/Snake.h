// Snake.h
#pragma once
#include <vector>
#include <raylib.h>

namespace game {

enum class Dir { UP, DOWN, LEFT, RIGHT };

struct Vec2i { 
    int x, y; 
    bool operator==(const Vec2i& other) const {
        return x == other.x && y == other.y;
    }
};

class Snake {
public:
    Snake(int gridW, int gridH);
    void reset();
    void setDirection(Dir d);
    std::tuple<float, float, float> getRayCast() const;
    bool update(); // returns false on collision
    const std::vector<Vec2i>& body() const;
    Vec2i head() const;
    void grow();
private:
    int gridW_, gridH_;
    std::vector<Vec2i> segments_;
    Dir dir_;
    bool growNext_;
};

} // namespace game
