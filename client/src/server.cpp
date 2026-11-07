// server.cpp — 单端口（8080）HTTP + WebSocket 服务：静态托管 + REST + 流式 Token
#define UNICODE
#include <winsock2.h>
#include <ws2tcpip.h>
#include <bcrypt.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <chrono>
#include "engine_host.h"
#include "tui.h"
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "bcrypt.lib")

static EngineHost g_engine;
static std::mutex g_cliMu;
static SOCKET g_client = INVALID_SOCKET;
static std::mutex g_loadMu;

// ---------- 工具 ----------
static std::string b64(const unsigned char* d, size_t n) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    for (size_t i = 0; i < n; i += 3) {
        unsigned v = d[i] << 16;
        if (i + 1 < n) v |= d[i + 1] << 8;
        if (i + 2 < n) v |= d[i + 2];
        o.push_back(T[(v >> 18) & 63]); o.push_back(T[(v >> 12) & 63]);
        o.push_back(i + 1 < n ? T[(v >> 6) & 63] : '=');
        o.push_back(i + 2 < n ? T[v & 63] : '=');
    }
    return o;
}
static std::string sha1(const std::string& in) {
    BCRYPT_ALG_HANDLE alg = nullptr; BCRYPT_HASH_HANDLE h = nullptr;
    BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA1_ALGORITHM, nullptr, 0);
    DWORD len = 0, got = 0;
    BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, (PUCHAR)&len, sizeof(len), &got, 0);
    BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0);
    BCryptHashData(h, (PUCHAR)in.data(), (ULONG)in.size(), 0);
    std::vector<unsigned char> o(len);
    BCryptFinishHash(h, o.data(), len, 0);
    BCryptDestroyHash(h); BCryptCloseAlgorithmProvider(alg, 0);
    return std::string((char*)o.data(), o.size());
}
static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') {}
        else if ((unsigned char)c < 0x20) o += ' ';
        else o.push_back(c);
    }
    return o;
}
static std::string frame(const std::string& s) {
    std::string o; o.push_back((char)0x81);
    const size_t n = s.size();
    if (n < 126) o.push_back((char)n);
    else { o.push_back((char)126); o.push_back((char)(n >> 8)); o.push_back((char)(n & 0xFF)); }
    o += s; return o;
}
static void ws_push(const std::string& json) {
    std::lock_guard<std::mutex> lk(g_cliMu);
    if (g_client != INVALID_SOCKET) { auto f = frame(json); ::send(g_client, f.data(), (int)f.size(), 0); }
}

static std::string telemetry_json(const std::string& sid = "") {
    std::lock_guard<std::mutex> lk(g_loadMu);
    auto t = g_engine.telemetry(sid);
    char b[512];
    std::snprintf(b, sizeof(b),
        "{\"ev\":\"tele\",\"tok_s\":%.2f,\"n_past\":%lld,\"vram_used\":%.2f,\"vram_cap\":%.2f,"
        "\"blocks_total\":%d,\"blocks_free\":%d,\"shared\":%d,\"cow\":%d,\"temp\":%.2f,\"topk\":%d}",
        t.tok_s, (long long)t.n_past, t.vram_used_gb, t.vram_cap_gb,
        t.blocks_total, t.blocks_free, t.blocks_shared, t.cow_count, t.k_temp, t.top_k);
    return b;
}

static std::string sessions_json() {
    std::string j = "{\"sessions\":[";
    bool first = true;
    for (auto& x : g_engine.sessions()) {
        if (!first) j += ","; first = false;
        j += "{\"id\":\"" + json_escape(x.id) + "\",\"title\":\"" + json_escape(x.title) +
             "\",\"n_past\":" + std::to_string(x.n_past) + ",\"turns\":" + std::to_string(x.turns) + "}";
    }
    return j + "]}";
}
static std::string history_json(const std::string& sid) {
    std::string j = "{\"messages\":[";
    bool first = true;
    for (auto& m : g_engine.history(sid)) {
        if (!first) j += ","; first = false;
        j += std::string("{\"role\":\"") + (m.user ? "user" : "assistant") +
             "\",\"text\":\"" + json_escape(m.text) + "\"}";
    }
    return j + "]}";
}
static std::string qparam(const std::string& path, const char* k) {
    const std::string key = std::string(k) + "=";
    size_t p = path.find(key);
    if (p == std::string::npos) return {};
    size_t a = p + key.size(), b = path.find_first_of("&", a);
    return path.substr(a, b == std::string::npos ? std::string::npos : b - a);
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
static const char* mime(const std::string& p) {
    if (p.size() > 5 && p.substr(p.size() - 5) == ".html") return "text/html; charset=utf-8";
    if (p.size() > 3 && p.substr(p.size() - 3) == ".js")   return "application/javascript; charset=utf-8";
    if (p.size() > 4 && p.substr(p.size() - 4) == ".css")  return "text/css; charset=utf-8";
    if (p.size() > 4 && p.substr(p.size() - 4) == ".svg")  return "image/svg+xml";
    return "text/plain; charset=utf-8";
}

static void http_reply(SOCKET c, int code, const char* ctype, const std::string& body) {
    std::string h = "HTTP/1.1 " + std::to_string(code) + (code == 200 ? " OK\r\n" : " Not Found\r\n");
    h += std::string("Content-Type: ") + ctype + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    ::send(c, h.data(), (int)h.size(), 0);
    if (!body.empty()) ::send(c, body.data(), (int)body.size(), 0);
}

static std::string field(const std::string& raw, const char* k) {
    // JSON: "k":"v"   或  form: k=v
    const std::string jk = std::string("\"") + k + "\"";
    size_t p = raw.find(jk);
    if (p != std::string::npos) {
        size_t c = raw.find(':', p);
        if (c != std::string::npos) {
            size_t a = raw.find('"', c);
            if (a != std::string::npos) {
                size_t b = raw.find('"', a + 1);
                if (b != std::string::npos) return raw.substr(a + 1, b - a - 1);
            }
        }
    }
    const std::string fk = std::string(k) + "=";
    p = raw.find(fk);
    if (p != std::string::npos) {
        size_t a = p + fk.size();
        size_t b = raw.find_first_of("&\r\n", a);
        return raw.substr(a, b == std::string::npos ? std::string::npos : b - a);
    }
    return {};
}

// ---------- WebSocket 会话 ----------
static void ws_loop(SOCKET c, const std::string& head) {
    const std::string tag = "Sec-WebSocket-Key: ";
    size_t p = head.find(tag);
    const std::string key = (p == std::string::npos) ? "" : head.substr(p + tag.size(), 24);
    const std::string acc = b64((const unsigned char*)sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").data(), 20);
    std::string resp = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n";
    resp += "Sec-WebSocket-Accept: " + acc + "\r\n\r\n";
    ::send(c, resp.data(), (int)resp.size(), 0);
    { std::lock_guard<std::mutex> lk(g_cliMu); g_client = c; }
    LOG_K("WebSocket 客户端已连接");

    char buf[1 << 16];
    for (;;) {
        int n = recv(c, buf, sizeof(buf), 0);
        if (n <= 0) break;
        int len = (unsigned char)buf[1] & 0x7F, off = 2;
        if (len == 126) { len = ((unsigned char)buf[2] << 8) | (unsigned char)buf[3]; off = 4; }
        const unsigned char* mask = (const unsigned char*)buf + off; off += 4;
        std::string msg(len, 0);
        for (int i = 0; i < len && off + i < n; ++i) msg[i] = (char)(buf[off + i] ^ mask[i & 3]);
        if (msg.find("chat") != std::string::npos) {
            std::string text = field(msg, "text");
            std::string sid = field(msg, "session_id");
            if (sid.empty()) sid = g_engine.new_session();
            ws_push("{\"ev\":\"start\",\"session_id\":\"" + json_escape(sid) + "\"}");
            g_engine.chat(sid, text);
        } else if (msg.find("fork") != std::string::npos) {
            const std::string sid = field(msg, "session_id");
            const int64_t fn = g_engine.fork(sid);
            LOG_K("fork %s -> %lld tokens", sid.c_str(), (long long)fn);
        }
    }
    { std::lock_guard<std::mutex> lk(g_cliMu); if (g_client == c) g_client = INVALID_SOCKET; }
    closesocket(c);
    LOG_K("WebSocket 客户端断开");
}

static void serve(SOCKET c) {
    const auto t0 = std::chrono::steady_clock::now();
    char buf[1 << 16];
    int n = recv(c, buf, sizeof(buf) - 1, 0);
    if (n <= 0) { closesocket(c); return; }
    buf[n] = 0;
    std::string head(buf);


    // 读全 body（Content-Length 可能跨包）
    {
        size_t bpos = head.find("\r\n\r\n");
        size_t cl = 0;
        size_t cp = head.find("Content-Length: ");
        if (cp != std::string::npos) cl = (size_t)atoi(head.c_str() + cp + 16);
        if (bpos != std::string::npos) {
            size_t have = head.size() - (bpos + 4);
            while (have < cl) {
                int m = recv(c, buf, sizeof(buf), 0);
                if (m <= 0) break;
                head.append(buf, m);
                have += (size_t)m;
            }
        }
    }

    if (head.find("Upgrade: websocket") != std::string::npos || head.find("Upgrade: WebSocket") != std::string::npos) {
        ws_loop(c, head); return;
    }

    // 请求行
    size_t sp1 = head.find(' ');
    size_t sp2 = head.find(' ', sp1 + 1);
    const std::string path = head.substr(sp1 + 1, sp2 - sp1 - 1);
    const bool post = head.rfind("POST", 0) == 0;

    if (path.rfind("/api/telemetry", 0) == 0) {
        http_reply(c, 200, "application/json", telemetry_json(qparam(path, "session_id")));
        closesocket(c); return;
    }
    if (path.rfind("/api/sessions", 0) == 0) {
        http_reply(c, 200, "application/json", sessions_json());
        closesocket(c); return;
    }
    if (path.rfind("/api/history", 0) == 0) {
        http_reply(c, 200, "application/json", history_json(qparam(path, "session_id")));
        closesocket(c); return;
    }
    if (path.rfind("/api/new", 0) == 0) {
        const std::string id = g_engine.new_session();
        http_reply(c, 200, "application/json", "{\"id\":\"" + json_escape(id) + "\"}");
        closesocket(c); return;
    }

    if (post && path == "/api/load") {
        size_t b = head.find("\r\n\r\n");
        std::string body = b == std::string::npos ? "" : head.substr(b + 4);
        std::string p = field(body, "path");
        std::string err; bool ok;
        LOG_G("加载模型: %s", p.c_str());
        {
            // 加载期间画进度条（引擎是黑盒，进度按耗时估算）
            std::atomic<bool> done{ false };
            std::thread anim([&done] {
                const char* notes[] = { "解析 GGUF 头…", "校验特征签名…", "分配 DeviceArena…",
                                        "上传张量至显存…", "构建 KV 块池…" };
                for (int pct = 0; pct < 96 && !done; pct += 2) {
                    tui_bar("LOADING", pct, notes[(pct / 20) % 5]);
                    std::this_thread::sleep_for(std::chrono::milliseconds(480));
                }
            });
            const auto lt0 = std::chrono::steady_clock::now();
            { std::lock_guard<std::mutex> lk(g_loadMu); ok = g_engine.load(p, err); }
            done = true; anim.join();
            const double s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - lt0).count();
            tui_bar("LOADING", 100, ok ? "完成" : "失败");
            std::printf("\n");
            if (ok) { LOG_G("模型就绪，用时 %.1f s", s_); const std::string nid = g_engine.new_session(); LOG_K("新会话 %s", nid.c_str()); }
        }
        std::string j = "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"err\":\"" + json_escape(err) + "\"}";
        http_reply(c, 200, "application/json", j); closesocket(c); return;
    }

    if (post && path == "/api/chat") {
        size_t b = head.find("\r\n\r\n");
        std::string body = b == std::string::npos ? "" : head.substr(b + 4);
        const std::string text = field(body, "text");
        std::string sid = field(body, "session_id");
        if (sid.empty()) sid = g_engine.new_session();
        ws_push("{\"ev\":\"start\",\"session_id\":\"" + json_escape(sid) + "\"}");
        g_engine.chat(sid, text);
        http_reply(c, 200, "application/json", "{\"ok\":true}"); closesocket(c); return;
    }

    // 静态文件
    std::string rel = (path == "/") ? "index.html" : path.substr(1);
    std::string file = "web/" + rel;
    std::string body = read_file(file);
    if (body.empty()) body = read_file("client/" + file);
    if (body.empty()) { http_reply(c, 404, "text/plain", "not found"); closesocket(c); return; }
    http_reply(c, 200, mime(rel), body);
    closesocket(c);
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    WSADATA w; WSAStartup(MAKEWORD(2, 2), &w);
    SOCKET ls = socket(AF_INET, SOCK_STREAM, 0);
    BOOL yes = TRUE; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(8080); a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(ls, (sockaddr*)&a, sizeof(a)) || listen(ls, 16)) { std::printf("bind 8080 failed\n"); return 1; }
    tui_init();
    tui_logo();
    LOG_G("引擎初始化完成");
    LOG_I("API 服务监听 :8080");
    LOG_I("浏览器打开 %shttp://localhost:8080%s", CLR_URL, CLR_RST);
    std::printf("\n");

    // 原地位移的运行状态栏
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            auto t = g_engine.telemetry("");
            std::printf("\r\033[2K");
            std::printf("\033[35m│\033[0m 显存池 \033[1m%.2f/%.2f\033[0m GB", t.vram_used_gb, t.vram_cap_gb);
            std::printf(" \033[35m│\033[0m 块 \033[1m%d/%d\033[0m", t.blocks_total - t.blocks_free, t.blocks_total);
            std::printf(" \033[35m│\033[0m 共享 \033[1;35m%d\033[0m", t.blocks_shared);
            std::printf(" \033[35m│\033[0m COW \033[1;35m%d\033[0m", t.cow_count);
            std::printf(" \033[35m│\033[0m \033[1;36m%.1f tok/s\033[0m", t.tok_s);
            std::printf(" \033[35m│\033[0m n_past \033[1m%lld\033[0m\033[0m", (long long)t.n_past);
            fflush(stdout);
        }
    }).detach();

    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            g_engine.pump([](const std::string& t) {
                ws_push("{\"ev\":\"tok\",\"t\":\"" + json_escape(t) + "\"}");
            });
            ws_push(telemetry_json());
        }
    }).detach();

    for (;;) {
        SOCKET c = accept(ls, nullptr, nullptr);
        if (c == INVALID_SOCKET) continue;
        std::thread([c] { serve(c); }).detach();
    }
}