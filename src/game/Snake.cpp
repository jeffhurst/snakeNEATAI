#include "Snake.h"
using namespace game;

Snake::Snake(int w, int h)
 : gridW_(w), gridH_(h), dir_(Dir::RIGHT), growNext_(false)
{
    reset();
}

void Snake::reset() {
    segments_.clear();
    segments_.push_back({gridW_/2, gridH_/2});
    dir_ = Dir::RIGHT;
    growNext_ = false;
}

void Snake::setDirection(Dir d) {
    // prevent reverse
    if ((dir_ == Dir::UP && d == Dir::DOWN) ||
        (dir_ == Dir::DOWN && d == Dir::UP) ||
        (dir_ == Dir::LEFT && d == Dir::RIGHT) ||
        (dir_ == Dir::RIGHT && d == Dir::LEFT))
        return;
    dir_ = d;
}

bool Snake::update() {
    Vec2i head = segments_.front();
    switch (dir_) {
        case Dir::UP:    head.y--; break;
        case Dir::DOWN:  head.y++; break;
        case Dir::LEFT:  head.x--; break;
        case Dir::RIGHT: head.x++; break;
    }
    // wall collision
    if (head.x < 0 || head.x >= gridW_ || head.y < 0 || head.y >= gridH_)
        return false;
    // self collision
    for (auto& s : segments_)
        if (s.x == head.x && s.y == head.y)
            return false;
    segments_.insert(segments_.begin(), head);
    if (growNext_) {
        growNext_ = false;
    } else {
        segments_.pop_back();
    }
    return true;
}

void Snake::grow() { growNext_ = true; }

const std::vector<Vec2i>& Snake::body() const { return segments_; }
Vec2i Snake::head() const { return segments_.front(); }
