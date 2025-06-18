// Snake.cpp
#include "Snake.h"
#include <iostream>
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



std::tuple<float, float, float> Snake::getRayCast() const {
    auto isBlocked = [&](Vec2i pos) {
        // Check wall
        if (pos.x < 0 || pos.x >= gridW_ || pos.y < 0 || pos.y >= gridH_)
            return true;
        // Check body
        for (const auto& seg : segments_)
            if (seg == pos)
                return true;
        return false;
    };

    // Directions relative to current
    Vec2i fwd, lft, rgt;
    switch (dir_) {
        case Dir::UP:    fwd = {0, -1}; lft = {-1, 0}; rgt = {1, 0}; break;
        case Dir::DOWN:  fwd = {0, 1};  lft = {1, 0};  rgt = {-1, 0}; break;
        case Dir::LEFT:  fwd = {-1, 0}; lft = {0, 1};  rgt = {0, -1}; break;
        case Dir::RIGHT: fwd = {1, 0};  lft = {0, -1}; rgt = {0, 1};  break;
    }

    auto checkDir = [&](Vec2i dirVec) -> float {
        Vec2i pos = head();
        for (int i = 1; i <= 3; ++i) {
            pos.x += dirVec.x;
            pos.y += dirVec.y;
            if (isBlocked(pos))
                return 1.0f - ((i - 1) / 3.0f);
        }
        return 0.0f;
    };

    float leftDist   = checkDir(lft);
    float frontDist  = checkDir(fwd);
    float rightDist  = checkDir(rgt);

    return {leftDist, frontDist, rightDist};
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
    if (head.x < 0 || head.x >= gridW_ || head.y < 0 || head.y >= gridH_){
        //std::cout << "wall" << std::endl;
        return false;
    }
    // self collision
    for (auto& s : segments_)
        if (s.x == head.x && s.y == head.y){
            //std::cout << "self" << std::endl;
            return false;
        }
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
