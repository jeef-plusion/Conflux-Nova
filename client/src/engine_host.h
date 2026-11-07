// engine_host.h — UI 与引擎的唯一接缝（同进程、零网络、零拷贝）
#pragma once
#include <string>
#include <functional>
#include <vector>
#include <cstdint>

struct Telemetry {
    double  tok_s = 0;
    int64_t n_past = 0;
    double  vram_used_gb = 0, vram_cap_gb = 0;
    int     blocks_total = 0, blocks_free = 0, blocks_shared = 0;
    int     cow_count = 0;
    float   k_temp = 0.8f;
    int     top_k = 40;
};

struct ForkNode { int64_t block; std::string label; };

class EngineHost {
public:
    using TokenFn = std::function<void(const std::string&)>;

    bool load(const std::string& gguf, std::string& err);

    // 异步：起推理线程，token 进队列；UI 每帧调用 pump() 取走
    void chat(const std::string& text);
    void pump(const TokenFn& on_token);
    bool busy() const;

    int64_t fork(int64_t block);     // 共享该块之前的前缀，refcnt++
    bool    attach(int64_t block);   // 把该块之前的前缀并入当前会话

    Telemetry telemetry() const;
    const std::vector<ForkNode>& tree() const;

private:
    Telemetry t_;
    std::vector<ForkNode> tree_;
};