#include "nova/loader/gguf.h"
#include <cstdio>
int main(int argc, char** argv) {
    nova::GgufFile g(argv[1]);
    for (const auto& t : g.tensors()) {
        std::printf("%-32s type=%2u ne=[%llu,%llu] bytes=%llu\n", t.name.c_str(), (unsigned)t.type,
                    (unsigned long long)(t.ne.size() > 0 ? t.ne[0] : 0),
                    (unsigned long long)(t.ne.size() > 1 ? t.ne[1] : 0),
                    (unsigned long long)t.bytes);
    }
    return 0;
}
