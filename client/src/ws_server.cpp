#include "ws_server.h"
#include <cstring>
#include <vector>

static const char* B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string WsServer::b64(const unsigned char* d, size_t n) {
    std::string o; o.reserve(((n + 2) / 3) * 4);
    for (size_t i = 0; i < n; i += 3) {
        unsigned v = d[i] << 16;
        if (i + 1 < n) v |= d[i + 1] << 8;
        if (i + 2 < n) v |= d[i + 2];
        o.push_back(B64[(v >> 18) & 63]); o.push_back(B64[(v >> 12) & 63]);
        o.push_back(i + 1 < n ? B64[(v >> 6) & 63] : '=');
        o.push_back(i + 2 < n ? B64[v & 63] : '=');
    }
    return o;
}

std::string WsServer::sha1(const std::string& in) {
    BCRYPT_ALG_HANDLE alg = nullptr; BCRYPT_HASH_HANDLE h = nullptr;
    BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA1_ALGORITHM, nullptr, 0);
    DWORD len = 0, got = 0;
    BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, (PUCHAR)&len, sizeof(len), &got, 0);
    BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0);
    BCryptHashData(h, (PUCHAR)in.data(), (ULONG)in.size(), 0);
    std::vector<unsigned char> out(len);
    BCryptFinishHash(h, out.data(), len, 0);
    BCryptDestroyHash(h); BCryptCloseAlgorithmProvider(alg, 0);
    return std::string((char*)out.data(), out.size());
}

std::string WsServer::frame(const std::string& s) {
    std::string o; o.push_back((char)0x81);
    const size_t n = s.size();
    if (n < 126) o.push_back((char)n);
    else if (n < 65536) { o.push_back((char)126); o.push_back((char)(n >> 8)); o.push_back((char)(n & 0xFF)); }
    else { o.push_back((char)127); for (int i = 7; i >= 0; --i) o.push_back((char)((n >> (i * 8)) & 0xFF)); }
    o += s; return o;
}

std::string WsServer::unframe(const char* b, int n) {
    if (n < 2) return {};
    int len = (unsigned char)b[1] & 0x7F; int off = 2;
    if (len == 126) { len = ((unsigned char)b[2] << 8) | (unsigned char)b[3]; off = 4; }
    else if (len == 127) { len = 0; for (int i = 0; i < 8; ++i) len = (len << 8) | (unsigned char)b[2 + i]; off = 10; }
    const unsigned char* mask = (const unsigned char*)b + off; off += 4;
    std::string s; s.resize((size_t)len);
    for (int i = 0; i < len && off + i < n; ++i) s[(size_t)i] = (char)(b[off + i] ^ mask[i & 3]);
    return s;
}

bool WsServer::start(unsigned short port, Handler h) {
    WSADATA w; if (WSAStartup(MAKEWORD(2, 2), &w)) return false;
    listen_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_ == INVALID_SOCKET) return false;
    BOOL yes = TRUE; setsockopt(listen_, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listen_, (sockaddr*)&a, sizeof(a)) || listen(listen_, 8)) return false;
    handler_ = std::move(h); run_ = true;
    std::thread([this] {
        while (run_) {
            SOCKET c = accept(listen_, nullptr, nullptr);
            if (c == INVALID_SOCKET) continue;
            { std::lock_guard<std::mutex> lk(mu_); client_ = c; }
            std::thread([this, c] { serve(c); }).detach();
        }
    }).detach();
    return true;
}

void WsServer::serve(SOCKET c) {
    char buf[1 << 16];
    int n = recv(c, buf, sizeof(buf) - 1, 0);
    if (n <= 0) { closesocket(c); return; }
    buf[n] = 0;
    std::string req(buf);
    const std::string keyTag = "Sec-WebSocket-Key: ";
    size_t p = req.find(keyTag);
    std::string key = (p != std::string::npos) ? req.substr(p + keyTag.size(), 24) : "";
    const std::string accept = b64((const unsigned char*)sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").data(), 20);
    std::string resp =
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
    ::send(c, resp.data(), (int)resp.size(), 0);

    for (;;) {
        n = recv(c, buf, sizeof(buf), 0);
        if (n <= 0) break;
        std::string m = unframe(buf, n);
        if (m.empty()) continue;
        if (handler_) handler_(m, [c](const std::string& s) { auto f = frame(s); ::send(c, f.data(), (int)f.size(), 0); });
    }
    { std::lock_guard<std::mutex> lk(mu_); if (client_ == c) client_ = INVALID_SOCKET; }
    closesocket(c);
}