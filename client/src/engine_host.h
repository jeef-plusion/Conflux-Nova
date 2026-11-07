// engine_host.h — 多会话：每个 session 独立 history / n_past / KV 块表
#pragma once
#include <string>
#include <vector>
#include <functional>
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

struct Msg { bool user; std::string text; };
struct SessionInfo { std::string id; std::string title; int64_t n_past; int turns; };

class EngineHost {
public:
    using TokenFn = std::function<void(const std::string&)>;

    bool load(const std::string& gguf, std::string& err);
    std::string new_session();
    bool chat(const std::string& sid, const std::string& text);
    void pump(const TokenFn& on_token);
    bool busy() const;

    int64_t fork(const std::string& sid);
    Telemetry telemetry(const std::string& sid) const;
    std::vector<SessionInfo> sessions() const;
    std::vector<Msg> history(const std::string& sid) const;

private:
    std::vector<SessionInfo> meta_;
};