// main.cpp — Win32 + D3D11 + Dear ImGui，单进程直连引擎
#define UNICODE
#include <windows.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <dxgi.h>
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "app.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

static ID3D11Device*           g_dev = nullptr;
static ID3D11DeviceContext*    g_ctx = nullptr;
static IDXGISwapChain*         g_swap = nullptr;
static ID3D11RenderTargetView* g_rtv = nullptr;
static App                     g_app;

static const wchar_t* kClass = L"ConfluxNovaNative";

static void apply_mica(HWND h) {
    const int backdrop = 2;                 // Mica
    DwmSetWindowAttribute(h, 38, &backdrop, sizeof(backdrop));
    const int corner = 2;                   // 圆角
    DwmSetWindowAttribute(h, 33, &corner, sizeof(corner));
    const int dark = 1;
    DwmSetWindowAttribute(h, 20, &dark, sizeof(dark));
    MARGINS m{ -1, -1, -1, -1 };
    DwmExtendFrameIntoClientArea(h, &m);
}

static bool init_d3d(HWND h) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = h;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    if (FAILED(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
            &sd, &g_swap, &g_dev, nullptr, &g_ctx))) return false;

    ID3D11Texture2D* back = nullptr;
    g_swap->GetBuffer(0, IID_PPV_ARGS(&back));
    g_dev->CreateRenderTargetView(back, nullptr, &g_rtv);
    back->Release();
    return true;
}

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (ImGui_ImplWin32_WndProcHandler(h, msg, wp, lp)) return true;
    switch (msg) {
    case WM_NCCALCSIZE: if (wp) return 0; break;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    case WM_SIZE:
        if (g_dev && wp != SIZE_MINIMIZED) {
            if (g_rtv) { g_rtv->Release(); g_rtv = nullptr; }
            g_swap->ResizeBuffers(0, LOWORD(lp), HIWORD(lp), DXGI_FORMAT_UNKNOWN, 0);
            ID3D11Texture2D* back = nullptr;
            g_swap->GetBuffer(0, IID_PPV_ARGS(&back));
            g_dev->CreateRenderTargetView(back, nullptr, &g_rtv);
            back->Release();
        }
        break;
    }
    return DefWindowProc(h, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    WNDCLASSEXW wc{ sizeof(wc) };
    wc.lpfnWndProc = wndproc; wc.hInstance = inst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = kClass;
    RegisterClassExW(&wc);

    HWND h = CreateWindowExW(0, kClass, L"Conflux Nova · FJEC_V2.2.0",
        WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_VISIBLE,
        140, 100, 1320, 840, nullptr, nullptr, inst, nullptr);
    apply_mica(h);
    if (!init_d3d(h)) return 1;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 10.f;
    ImGui::GetStyle().FrameRounding = 8.f;
    ImGui::GetStyle().Colors[ImGuiCol_WindowBg] = ImVec4(0.043f, 0.059f, 0.078f, 0.86f);
    ImGui_ImplWin32_Init(h);
    ImGui_ImplDX11_Init(g_dev, g_ctx);

    MSG msg{};
    for (;;) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg); DispatchMessage(&msg);
            if (msg.message == WM_QUIT) return 0;
        }
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        g_app.frame();                       // ← UI 直接读引擎内存
        ImGui::Render();
        const float clear[4] = { 0.043f, 0.059f, 0.078f, 0.86f };
        g_ctx->OMSetRenderTargets(1, &g_rtv, nullptr);
        g_ctx->ClearRenderTargetView(g_rtv, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_swap->Present(1, 0);
    }
}