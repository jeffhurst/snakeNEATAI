#pragma once
#include <vector>
#include <memory>
#include <mutex>

namespace core {

template<typename T>
class PoolManager {
public:
    // Acquire an object from pool or create new
    template<typename... Args>
    T* acquire(Args&&... args) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!pool_.empty()) {
            T* obj = pool_.back();
            pool_.pop_back();
            new (obj) T(std::forward<Args>(args)...);
            return obj;
        } else {
            return new T(std::forward<Args>(args)...);
        }
    }
    // Return object to pool
    void release(T* obj) {
        std::lock_guard<std::mutex> lk(mutex_);
        obj->~T();
        pool_.push_back(obj);
    }
    ~PoolManager() {
        for (T* p : pool_) delete p;
    }
private:
    std::vector<T*> pool_;
    std::mutex mutex_;
};

} // namespace core
