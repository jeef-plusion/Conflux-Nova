// build/lt/naked_malloc.cu - the naked test: a fresh process, first thing, one cudaMalloc.
//
// Why: if a brand-new process can take 4.5 GiB in ONE call, then there is no 2 GiB hard
// lock - our DeviceArena (which asks in 512 MiB chunks) was fighting fragmentation or an
// allocation-count limit, exactly as llama.cpp-style "one big block up front" avoids.
//
// usage: naked_malloc.exe <GiB> [--touch]
//   allocates exactly that much, once, then (optionally) writes first/last 1 MiB.
//
// ASCII only.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const double gib = (argc > 1) ? std::atof(argv[1]) : 4.5;
    const bool touch = (argc > 2 && std::strcmp(argv[2], "--touch") == 0);

    std::size_t f0 = 0, t0 = 0;
    cudaMemGetInfo(&f0, &t0);
    std::printf("fresh process: cuda free %.2f / %.2f GiB, asking %.2f GiB in ONE call\n",
                f0 / 1073741824.0, t0 / 1073741824.0, gib);

    const std::size_t bytes = (std::size_t)(gib * 1073741824.0);
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, bytes);
    std::printf("  cudaMalloc(%.2f GiB) -> %s", gib, e == cudaSuccess ? "OK" : cudaGetErrorString(e));
    if (e != cudaSuccess) {
        std::size_t f1 = 0, t1 = 0;
        cudaMemGetInfo(&f1, &t1);
        std::printf("   (free still %.2f GiB)\n", f1 / 1073741824.0);
        return 1;
    }
    std::printf("   ptr=%p\n", p);

    if (touch) {
        // write both ends: proves the whole range is really backed, not just reserved
        const cudaError_t a = cudaMemset(p, 0x7E, 1 << 20);
        const cudaError_t b = cudaMemset((char*)p + bytes - (1 << 20), 0x3C, 1 << 20);
        unsigned char probe[16] = {};
        cudaMemcpy(probe, p, sizeof(probe), cudaMemcpyDeviceToHost);
        unsigned char end[16] = {};
        cudaMemcpy(end, (char*)p + bytes - sizeof(end), sizeof(end), cudaMemcpyDeviceToHost);
        std::printf("  touch: head memset=%s (0x%02X) tail memset=%s (0x%02X)\n",
                    a == cudaSuccess ? "OK" : cudaGetErrorString(a), probe[0],
                    b == cudaSuccess ? "OK" : cudaGetErrorString(b), end[0]);
        // a real bandwidth read over the whole thing, so we know it is not a phantom mapping
        const auto t = clock();
        cudaMemset(p, 0, bytes);
        cudaDeviceSynchronize();
        const double ms = (double)(clock() - t) * 1000.0 / CLOCKS_PER_SEC;
        std::printf("  full memset %.2f GiB in %.1f ms -> %.1f GB/s (host clock, approximate)\n",
                    gib, ms, gib * 1.073741824 / (ms * 1e-3));
    }

    cudaFree(p);
    std::printf("  freed\n");
    return 0;
}
