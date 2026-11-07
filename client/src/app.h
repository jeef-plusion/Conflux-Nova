// app.h — 纯 ImGui 三栏界面；所有数据来自 C++ 内存，没有网络
#pragma once
#include "imgui.h"
#include "engine_host.h"
#include <deque>
#include <vector>
#include <string>
#include <vector>

struct App {
    EngineHost engine;

    struct Msg { bool user; std::string text; int64_t block = -1; };
    std::vector<Msg> chat;
    std::vector<float> tps = std::vector<float>(256, 0.f);
    char input[4096] = {};
    char gguf[512] = "build/q4k.gguf";
    std::string status = "未加载";
    int pending = -1;   // 正在接收 token 的气泡下标
    bool show_right = true;
    Telemetry t;

    void frame() {
        t = engine.telemetry();
        tps.erase(tps.begin()); tps.push_back((float)t.tok_s);

        // ← 每帧把推理线程压进来的 token 追加到气泡（无锁读队列）
        engine.pump([this](const std::string& tok) {
            if (pending >= 0 && pending < (int)chat.size()) chat[pending].text += tok;
        });

        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("##root", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground);

        const float H = io.DisplaySize.y;
        const float W = io.DisplaySize.x;
        const float leftW = 260.f, rightW = show_right ? 300.f : 0.f;

        ImGui::BeginChild("##left", ImVec2(leftW, H), true);
        drawLeft();
        ImGui::EndChild();
        ImGui::SameLine(0, 0);

        ImGui::BeginChild("##center", ImVec2(W - leftW - rightW, H), true);
        drawCenter();
        ImGui::EndChild();

        if (show_right) {
            ImGui::SameLine(0, 0);
            ImGui::BeginChild("##right", ImVec2(rightW, H), true);
            drawRight();
            ImGui::EndChild();
        }

        ImGui::End();
        ImGui::PopStyleVar();
    }

private:
    void drawLeft() {
        ImGui::TextColored(ImVec4(0.35f, 0.66f, 0.90f, 1.f), "CONFLUX NOVA");
        ImGui::TextDisabled("FJEC_V2.2.0-Core");
        ImGui::Spacing();
        ImGui::InputText("##gguf", gguf, sizeof(gguf));
        if (ImGui::Button("点火", ImVec2(-1, 30))) {
            std::string err;
            status = engine.load(gguf, err) ? "已加载" : ("失败: " + err);
        }
        ImGui::Separator();
        ImGui::TextDisabled("会话");
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 60);
        if (ImGui::SmallButton("+ New")) chat.clear();

        for (auto& n : engine.tree()) {
            ImGui::PushID((int)n.block);
            if (ImGui::Selectable(n.label.c_str())) {}
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Fork")) engine.fork(n.block);
                if (ImGui::MenuItem("Merge")) engine.attach(n.block);
                ImGui::EndPopup();
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Fork")) engine.fork(n.block);
            ImGui::PopID();
        }

        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 44);
        ImGui::Separator();
        ImGui::TextDisabled("GTX 1080 Ti · 11 GB");
        ImGui::TextDisabled("%s", status.c_str());
    }

    void drawCenter() {
        ImGui::TextDisabled("对话"); ImGui::SameLine(ImGui::GetContentRegionAvail().x - 90);
        if (ImGui::SmallButton(show_right ? "隐藏遥测" : "显示遥测")) show_right = !show_right;
        ImGui::Separator();

        ImGui::BeginChild("##log", ImVec2(0, -84), false);
        for (auto& m : chat) {
            ImGui::PushStyleColor(ImGuiCol_Text, m.user
                ? ImVec4(0.55f, 0.78f, 0.98f, 1.f) : ImVec4(0.90f, 0.93f, 0.96f, 1.f));
            ImGui::TextWrapped("%s%s", m.user ? "你: " : "AI: ", m.text.c_str());
            ImGui::PopStyleColor();
            if (!m.user && m.block >= 0) {
                ImGui::SameLine();
                if (ImGui::SmallButton(("Fork##" + std::to_string(m.block)).c_str())) engine.fork(m.block);
            }
            ImGui::Spacing();
        }
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4) ImGui::SetScrollHereY(1.f);
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::InputTextMultiline("##in", input, sizeof(input), ImVec2(-90, 64));
        ImGui::SameLine();
        if (ImGui::Button("发送", ImVec2(80, 64))) send();
        if (ImGui::IsItemHovered() && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) send();
    }

    void drawRight() {
        ImGui::TextColored(ImVec4(0.35f, 0.66f, 0.90f, 1.f), "Engine Telemetry");
        ImGui::Separator();
        ImGui::Text("吞吐 %.1f tok/s", t.tok_s);
        ImGui::PlotLines("##tps", tps.data(), (int)tps.size(), 0, nullptr, 0.f, 140.f, ImVec2(-1, 70));
        ImGui::Text("显存池 %.2f / %.2f GB", t.vram_used_gb, t.vram_cap_gb);
        ImGui::ProgressBar((float)(t.vram_used_gb / (t.vram_cap_gb > 0 ? t.vram_cap_gb : 1)),
                           ImVec2(-1, 6), "");
        ImGui::Text("物理块 %d / %d", t.blocks_total - t.blocks_free, t.blocks_total);
        ImGui::Text("共享块 %d", t.blocks_shared);
        ImGui::TextColored(ImVec4(0.89f, 0.39f, 0.35f, 1.f), "COW 触发 %d", t.cow_count);
        ImGui::Separator();
        ImGui::TextDisabled("kTemp  %.2f", t.k_temp);
        ImGui::TextDisabled("Top-K  %d", t.top_k);
        ImGui::TextDisabled("n_past %lld", (long long)t.n_past);
    }

    void send() {
        if (!input[0]) return;
        chat.push_back({ true, input, -1 });
        chat.push_back({ false, "", (int64_t)chat.size() });
        pending = (int)chat.size() - 1;
        std::string text = input; input[0] = 0;
        engine.chat(text);          // 异步：立刻返回，token 由 frame() 的 pump 取走
    }
};