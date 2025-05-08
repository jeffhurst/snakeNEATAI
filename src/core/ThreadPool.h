#pragma once
#include <vector>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace core {

class ThreadPool {
public:
    explicit ThreadPool(size_t n) : done_(false) {
        for (size_t i = 0; i < n; ++i)
            workers_.emplace_back(&ThreadPool::workerLoop, this);
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            done_ = true;
        }
        cond_.notify_all();
        for (auto& w : workers_) w.join();
    }
    void enqueue(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            jobs_.push(std::move(job));
        }
        cond_.notify_one();
    }
private:
    void workerLoop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cond_.wait(lk, [&]{ return done_ || !jobs_.empty(); });
                if (done_ && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop();
            }
            job();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> jobs_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool done_;
};

} // namespace core
