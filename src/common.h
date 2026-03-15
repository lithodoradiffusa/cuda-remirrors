#pragma once

#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <array>

#ifndef MIRRORS_LARGE_BIOMES
#define MIRRORS_LARGE_BIOMES 0
#endif
#if MIRRORS_LARGE_BIOMES
constexpr bool large_biomes = true;
#else
constexpr bool large_biomes = false;
#endif

struct GpuOutput {
    uint64_t seed;
    int32_t x;
    int32_t z;
};

struct CpuOutput {
    uint64_t seed;
    int32_t x;
    int32_t z;
    int32_t score;
};

struct GpuOutputs {
    std::queue<GpuOutput> queue;
    std::mutex mutex;
};

struct CpuOutputs {
    std::queue<CpuOutput> queue;
    std::mutex mutex;
};


template<typename T>
struct Thread {
private:
    std::atomic_bool stop_flag;
    std::thread thread;

protected:
    Thread() : stop_flag(false), thread() {

    }

    void start() {
        thread = std::thread(&T::run, (T*)this);
    }

    bool should_stop() {
        return stop_flag.load(std::memory_order_relaxed);
    }

public:
    void stop() {
        stop_flag.store(true, std::memory_order_relaxed);
    }

    void join() {
        thread.join();
    }
};
