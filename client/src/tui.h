// tui.h — 终端 UI：ANSI 颜色、日志前缀、进度条、原地位移状态栏（零依赖）
#pragma once
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <string>
#include <windows.h>

#define CLR_RST  "\033[0m"
#define CLR_DIM  "\033[2m"
#define CLR_INFO "\033[32m"
#define CLR_WARN "\033[33m"
#define CLR_ERR  "\033[1;31m"
#define CLR_ENG  "\033[36m"
#define CLR_KV   "\033[35m"
#define CLR_URL  "\033[1;36m"
#define CLR_HOT  "\033[1;35m"

inline void tui_init() {
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(h, &mode)) SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT);
    SetConsoleOutputCP(CP_UTF8);
}

inline std::string tui_time() {
    SYSTEMTIME t; GetLocalTime(&t);
    char b[16]; std::snprintf(b, sizeof(b), "%02d:%02d:%02d", t.wHour, t.wMinute, t.wSecond);
    return b;
}

inline void tui_log(const char* color, const char* tag, const char* fmt, ...) {
    printf("\r\033[2K%s[%s]%s %s[%s]%s ", CLR_DIM, tui_time().c_str(), CLR_RST,
           color, tag, CLR_RST);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n");
}

#define LOG_I(...) tui_log(CLR_INFO, "INFO",   __VA_ARGS__)
#define LOG_W(...) tui_log(CLR_WARN, "WARN",   __VA_ARGS__)
#define LOG_E(...) tui_log(CLR_ERR,  "ERROR",  __VA_ARGS__)
#define LOG_G(...) tui_log(CLR_ENG,  "ENGINE", __VA_ARGS__)
#define LOG_K(...) tui_log(CLR_KV,   "KV",     __VA_ARGS__)
#define LOG_H(...) tui_log(CLR_URL,  "HTTP",   __VA_ARGS__)

inline void tui_logo() {
    const char* art[] = {
        "",
        "   ██████╗ ██████╗ ███╗   ██╗███████╗██╗     ██╗   ██╗██╗  ██╗",
        "  ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║     ██║   ██║╚██╗██╔╝",
        "  ██║     ██║   ██║██╔██╗ ██║█████╗  ██║     ██║   ██║ ╚███╔╝ ",
        "  ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║     ██║   ██║ ██╔██╗ ",
        "  ╚██████╗╚██████╔╝██║ ╚████║██║     ███████╗╚██████╔╝██╔╝ ██╗",
        "   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝",
        "",
        "              N O V A   ·   FJEC_V2.2.0-Core",
        "",
    };
    for (const char* l : art) printf("\033[1;35m%s\033[0m\n", l);
}

inline void tui_bar(const char* label, int pct, const char* note = "") {
    const int W = 32;
    const int f = pct * W / 100;
    printf("\r\033[2K  %s%s%s ", CLR_ENG, label, CLR_RST);
    printf("\033[35m[");
    for (int i = 0; i < W; ++i) printf(i < f ? "█" : "░");
    printf("]\033[0m \033[1m%3d%%\033[0m  %s%s%s", pct, CLR_DIM, note, CLR_RST);
    fflush(stdout);
}
