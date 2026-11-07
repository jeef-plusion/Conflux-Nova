// main.cpp — Win32 + Mica + WebView2；本地 WS 直连引擎
#define UNICODE
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>      // 必须先于 windows.h，否则拉进 winsock1 冲突
#include <ws2tcpip.h>
#include <windows.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <shlobj.h>
#include <wrl.h>
#include <string>
#include <thread>
#include <chrono>
#include "WebView2.h"
#include "ws_server.h"
#include "engine_host.h"
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "ole32.lib")

using namespace Microsoft::WRL;
static const wchar_t* kClass = L"ConfluxNovaWV";
static ComPtr<ICoreWebView2Controller> g_ctrl;
static ComPtr<ICoreWebView2> g_web;
static WsServer g_ws;
static EngineHost g_engine;
static bool g_loaded = false;

static void apply_mica(HWND h) {
    const int backdrop = 2, corner = 2, dark = 1;
    DwmSetWindowAttribute(h, 38, &backdrop, sizeof(backdrop));
    DwmSetWindowAttribute(h, 33, &corner, sizeof(corner));
    DwmSetWindowAttribute(h, 20, &dark, sizeof(dark));
    MARGINS m{ -1, -1, -1, -1 };
    DwmExtendFrameIntoClientArea(h, &m);
}

static std::string json_escape(const std::string& s) {
    std::string o; for (char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') {}
        else if ((unsigned char)c < 0x20) o += ' ';
        else o.push_back(c);
    } return o;
}

static std::string telemetry_json() {
    auto t = g_engine.telemetry();
    char b[512];
    std::snprintf(b, sizeof(b),
        "{\"ev\":\"tele\",\"tok_s\":%.2f,\"n_past\":%lld,\"vram_used\":%.2f,\"vram_cap\":%.2f,"
        "\"blocks_total\":%d,\"blocks_free\":%d,\"shared\":%d,\"cow\":%d,\"temp\":%.2f,\"topk\":%d}",
        t.tok_s, (long long)t.n_past, t.vram_used_gb, t.vram_cap_gb,
        t.blocks_total, t.blocks_free, t.blocks_shared, t.cow_count, t.k_temp, t.top_k);
    return b;
}

static void on_cmd(const std::string& raw, std::function<void(const std::string&)> reply) {
    auto has = [&](const char* k) { return raw.find(std::string("\"") + k + "\"") != std::string::npos; };
    if (has("load")) {
        std::string path = "build/q4k.gguf";
        size_t p = raw.find("\"path\"");
        if (p != std::string::npos) {
            size_t a = raw.find('"', raw.find(':', p));
            size_t b = raw.find('"', a + 1);
            if (a != std::string::npos && b != std::string::npos) path = raw.substr(a + 1, b - a - 1);
        }
        std::string err;
        g_loaded = g_engine.load(path, err);
        reply("{\"ev\":\"loaded\",\"ok\":" + std::string(g_loaded ? "true" : "false") +
              ",\"err\":\"" + json_escape(err) + "\"}");
        reply(telemetry_json());
        return;
    }
    if (has("chat")) {
        size_t p = raw.find("\"text\"");
        std::string text = "";
        if (p != std::string::npos) {
            size_t a = raw.find('"', raw.find(':', p));
            size_t b = raw.rfind('"');
            if (a != std::string::npos && b > a) text = raw.substr(a + 1, b - a - 1);
        }
        g_engine.chat(text);
        reply("{\"ev\":\"start\"}");
        return;
    }
    if (has("fork")) {
        const auto& tr = g_engine.tree();
        if (!tr.empty()) g_engine.fork(tr.back().block);
        reply(telemetry_json());
        return;
    }
    reply("{}");
}

static void init_webview(HWND hwnd) {
    wchar_t dir[MAX_PATH];
    SHGetFolderPathW(nullptr, CSIDL_LOCAL_APPDATA, nullptr, 0, dir);
    std::wstring data = std::wstring(dir) + L"\\ConfluxNova\\WV2";

    CreateCoreWebView2EnvironmentWithOptions(nullptr, data.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [hwnd](HRESULT, ICoreWebView2Environment* env) -> HRESULT {
                env->CreateCoreWebView2Controller(hwnd,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [](HRESULT, ICoreWebView2Controller* c) -> HRESULT {
                            g_ctrl = c; g_ctrl->get_CoreWebView2(&g_web);
                            // 背景透明（需要 Controller2）
                            ComPtr<ICoreWebView2Controller2> c2;
                            if (SUCCEEDED(g_ctrl->QueryInterface(IID_PPV_ARGS(c2.GetAddressOf())))) {
                                COREWEBVIEW2_COLOR bg{ 0, 0, 0, 0 };
                                c2->put_DefaultBackgroundColor(bg);
                            }
                            // 虚拟站点映射（需要 WebView2_3）
                            ComPtr<ICoreWebView2_3> w3;
                            if (SUCCEEDED(g_web->QueryInterface(IID_PPV_ARGS(w3.GetAddressOf())))) {
                                w3->SetVirtualHostNameToFolderMapping(
                                    L"app.local", L"web", COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_ALLOW);
                            }
                            g_web->Navigate(L"https://app.local/index.html");
                            return S_OK;
                        }).Get());
                return S_OK;
            }).Get());
}

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_NCCALCSIZE: if (wp) return 0; break;
    case WM_SIZE: if (g_ctrl) { RECT rc; GetClientRect(h, &rc); g_ctrl->put_Bounds(rc); } break;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProc(h, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    WNDCLASSEXW wc{ sizeof(wc) };
    wc.lpfnWndProc = wndproc; wc.hInstance = inst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = kClass;
    RegisterClassExW(&wc);

    HWND h = CreateWindowExW(0, kClass, L"Conflux Nova · FJEC_V2.2.0",
        WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_VISIBLE,
        140, 100, 1320, 840, nullptr, nullptr, inst, nullptr);
    apply_mica(h);

    g_ws.start(17530, on_cmd);
    init_webview(h);

    // 每 250ms 推一次遥测
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            g_engine.pump([](const std::string& tok) {
                g_ws.push("{\"ev\":\"tok\",\"t\":\"" + json_escape(tok) + "\"}");
            });
            g_ws.push(telemetry_json());
        }
    }).detach();

    MSG m; while (GetMessage(&m, nullptr, 0, 0)) { TranslateMessage(&m); DispatchMessage(&m); }
    return 0;
}