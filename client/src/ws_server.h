// ws_server.h — 本地 WebSocket（127.0.0.1），正确实现握手
#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>
#include <bcrypt.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <functional>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "bcrypt.lib")

class WsServer {
public:
    using Handler = std::function<void(const std::string&, std::function<void(const std::string&)>)>;

    bool start(unsigned short port, Handler h);
    void push(const std::string& json) {
        std::lock_guard<std::mutex> lk(mu_);
        if (client_ != INVALID_SOCKET) { auto f = frame(json); ::send(client_, f.data(), (int)f.size(), 0); }
    }

private:
    static std::string b64(const unsigned char*, size_t);
    static std::string sha1(const std::string&);
    static std::string frame(const std::string&);
    static std::string unframe(const char*, int);
    void serve(SOCKET);

    SOCKET listen_ = INVALID_SOCKET, client_ = INVALID_SOCKET;
    std::atomic<bool> run_{ false };
    std::mutex mu_;
    Handler handler_;
};