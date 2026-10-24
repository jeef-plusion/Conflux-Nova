// nova_v2/src/main.cpp — 启动即校验：签名对不上就死，变得只是名字则放行并告警
#include <cstdio>
#include "nova/kernel.h"
#include "nova/loader/gguf.h"
#include "nova/loader/validate.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "用法: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    std::printf("[nova] 模具: %s | L=%d D=%d NH=%d NKV=%d HD=%d FFN=%d V=%d theta=%d tied=%d bias=%d\n",
                kModelName, nova::Model::L, nova::Model::D, nova::Model::NH, nova::Model::NKV,
                nova::Model::HD, nova::Model::FFN, nova::Model::V, nova::Model::theta,
                (int)nova::Model::tied, (int)nova::Model::qkv_bias);
    std::printf("[nova] 预算: 权重 %.3f GiB (上限 7.000) | KV@%d %.3f GiB\n",
                nova::Model::weight_bytes() / 1073741824.0, nova::kCtx,
                nova::Model::kv_bytes() / 1073741824.0);

    nova::GgufFile g(argv[1]);
    const nova::Verification v = nova::verify_model(g);

    std::printf("[nova] 校验通过: \"%s\"%s | %u 个 Q4_K 权重 / %u 个 norm | %.3f GiB\n",
                v.name.c_str(), v.variant ? "（变种，按模具执行）" : "",
                v.n_weight, v.n_norm, v.weight_bytes / 1073741824.0);
    return 0;
}
