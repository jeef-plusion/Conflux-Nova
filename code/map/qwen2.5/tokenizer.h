// code/map/qwen2.5/tokenizer.h — GGUF 词表 + 字节级 BPE
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <climits>
#include "nova/loader/gguf.h"

namespace nova {

inline void utf8_put(std::string& o, uint32_t cp) {
    if (cp < 0x80) o.push_back((char)cp);
    else if (cp < 0x800) { o.push_back((char)(0xC0 | (cp >> 6))); o.push_back((char)(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { o.push_back((char)(0xE0 | (cp >> 12))); o.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); o.push_back((char)(0x80 | (cp & 0x3F))); }
    else { o.push_back((char)(0xF0 | (cp >> 18))); o.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
           o.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); o.push_back((char)(0x80 | (cp & 0x3F))); }
}

class Tokenizer {
public:
    void load(const GgufFile& g) {
        const auto& tk = g.str_array("tokenizer.ggml.tokens");
        tokens_ = tk;
        tok2id_.reserve(tk.size() * 2);
        for (int i = 0; i < (int)tk.size(); ++i) tok2id_.emplace(tk[i], i);
        const auto& mg = g.str_array("tokenizer.ggml.merges");
        merge_.reserve(mg.size() * 2);
        for (int i = 0; i < (int)mg.size(); ++i) merge_.emplace(mg[i], i);
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        size_t i = 0;
        while (i < text.size()) {
            size_t j = i;
            if (text[j] == ' ') ++j;                       // 前导空格跟随后面的词
            while (j < text.size() && text[j] != ' ') ++j;
            bpe_word(text.substr(i, j - i), out);
            i = j;
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string bytes;
        for (int id : ids) {
            if (id < 0 || id >= (int)tokens_.size()) continue;
            for (unsigned char c : tokens_[id]) {
                auto it = uni2byte_.find(std::string(1, (char)c));
                (void)it;                                   // 多字节映射在下面按完整字符处理
            }
            // 逐字符（UTF-8 码点）回映射
            const std::string& s = tokens_[id];
            for (size_t k = 0; k < s.size();) {
                const int n = utf8_len((unsigned char)s[k]);
                auto it = uni2byte_.find(s.substr(k, n));
                if (it != uni2byte_.end()) bytes.push_back((char)it->second);
                else bytes.append(s.substr(k, n));
                k += n;
            }
        }
        return bytes;
    }

    int id(const std::string& tok, int dflt = -1) const {
        auto it = tok2id_.find(tok);
        return it == tok2id_.end() ? dflt : it->second;
    }

private:
    static int utf8_len(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4; }

    static const std::vector<std::string>& byte2uni() {
        static const std::vector<std::string> t = [] {
            std::vector<std::string> v(256);
            int n = 0;
            for (int b = 0; b < 256; ++b) {
                const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
                std::string s;
                if (keep) { utf8_put(s, (uint32_t)b); }        // 码点 → UTF-8（高位字节是两字节）
                else      { utf8_put(s, 256u + (uint32_t)n); ++n; }
                v[b] = s;
            }
            return v;
        }();
        return t;
    }

    void init_rev() {
        if (!uni2byte_.empty()) return;
        const auto& t = byte2uni();
        for (int b = 0; b < 256; ++b) uni2byte_[t[b]] = (uint8_t)b;
    }

    void bpe_word(const std::string& w, std::vector<int>& out) const {
        init_rev_const();
        std::vector<std::string> sym;
        sym.reserve(w.size());
        for (unsigned char c : w) sym.push_back(byte2uni()[c]);
        while (sym.size() > 1) {
            int best = INT_MAX, bi = -1;
            for (size_t i = 0; i + 1 < sym.size(); ++i) {
                auto it = merge_.find(sym[i] + " " + sym[i + 1]);
                if (it != merge_.end() && it->second < best) { best = it->second; bi = (int)i; }
            }
            if (bi < 0) break;
            sym[bi] += sym[bi + 1];
            sym.erase(sym.begin() + bi + 1);
        }
        for (const auto& s : sym) {
            auto it = tok2id_.find(s);
            if (it != tok2id_.end()) out.push_back(it->second);
        }
    }

    void init_rev_const() const { const_cast<Tokenizer*>(this)->init_rev(); }

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int> tok2id_;
    std::unordered_map<std::string, int> merge_;
    mutable std::unordered_map<std::string, uint8_t> uni2byte_;
};

} // namespace nova
