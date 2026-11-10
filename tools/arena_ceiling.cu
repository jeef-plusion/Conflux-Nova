// build/lt/arena_ceiling.cu - what can this process actually get from the driver, right now?
//
// Answers two numbers that WDDM reports inconsistently:
//   MAXSINGLE: the largest single cudaMalloc that succeeds (binary search, 256 MiB steps)
//   MAXTOTAL : how much can be held at once using 256 MiB chunks
// Both in a fresh process, nothing else allocated. Run it before/after rebooting or with
// ToDesk stopped; if the numbers move, the ceiling is environmental (remote capture /
// driver state), not a property of our engine.
//
// usage: arena_ceiling.exe
// ASCII only.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace {
constexpr std::size_t kMiB = 1024ull * 1024ull;
constexpr std::size_t kStep = 256ull * kMiB;

bool try_alloc(std::size_t bytes) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, bytes);
    if (e == cudaSuccess) { cudaFree(p); return true; }
    cudaGetLastError();
    return false;
}
} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    std::printf("=== allocation ceiling probe ===\n");
    std::printf("cudaMemGetInfo: free %.2f / total %.2f GiB\n", f / 1073741824.0, t / 1073741824.0);

    // ---- MAXSINGLE: largest single block, binary search on 256 MiB steps ----
    std::size_t lo = 0, hi = f;                 // free is an upper bound
    if (!try_alloc(kStep)) {
        std::printf("MAXSINGLE: even 256 MiB fails\n");
    } else {
        std::size_t a = kStep, b = hi;          // a works, b is unknown
        while (b - a > kStep) {
            const std::size_t mid = (a + b) / 2 / kStep * kStep;
            if (mid <= a) break;
            if (try_alloc(mid)) a = mid;
            else                b = mid;
        }
        lo = a;
        std::printf("MAXSINGLE: %.2f GiB (%.0f MiB)\n", lo / 1073741824.0, lo / (double)kMiB);
    }

    // ---- MAXTOTAL: how much can be held with 256 MiB chunks ----
    std::vector<void*> keep;
    std::size_t total = 0;
    for (int i = 0; i < 64; ++i) {
        void* p = nullptr;
        if (cudaMalloc(&p, kStep) != cudaSuccess) { cudaGetLastError(); break; }
        keep.push_back(p);
        total += kStep;
    }
    std::printf("MAXTOTAL : %.2f GiB in %zu x 256 MiB chunks\n", total / 1073741824.0, keep.size());
    for (void* p : keep) cudaFree(p);

    // ---- verdict hint ----
    std::printf("\n");
    if (lo >= 4ull * 1024 * kMiB) {
        std::printf("VERDICT: a single 4 GiB+ block IS available - 7B can be loaded with\n"
                    "         one big cudaMalloc (llama.cpp style). Fix the arena to try that first.\n");
    } else if (total > lo + kStep) {
        std::printf("VERDICT: single block is capped at %.2f GiB but %.2f GiB total works ->\n"
                    "         chunked allocation is the right shape; the cap is per-allocation.\n",
                    lo / 1073741824.0, total / 1073741824.0);
    } else {
        std::printf("VERDICT: both capped around %.2f GiB while cudaMemGetInfo reports %.2f GiB free.\n"
                    "         That gap is the environmental ceiling (remote capture / driver state).\n"
                    "         Re-run this tool after a reboot and with ToDesk stopped to compare.\n",
                    total / 1073741824.0, f / 1073741824.0);
    }
    return 0;
}
