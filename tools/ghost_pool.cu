// build/lt/ghost_pool.cu - FJEC_V2.4.0 ghost-process pool / VMM remap POC (Windows + WDDM)
//
// Three independent experiments; each fails or passes on its own:
//   E1 ghost processes : main holds 1.5 GB, then N children each hold 1.5 GB.
//                        If all children succeed, the WDDM budget is per-PID.
//   E2 CUDA IPC        : child exports cudaIpcGetMemHandle, parent imports.
//                        (Expected: unsupported on Windows. We measure the error code.)
//   E3 VMM remap       : cuMemAddressReserve 4.5 GB of VA (no physical), then
//                        cuMemCreate + cuMemMap physical chunks into it, memset the
//                        whole VA range and read it back. VMM gives contiguity, not
//                        extra physical memory - the experiment shows which.
//
// usage: ghost_pool.exe            run E1/E2/E3
//        ghost_pool.exe --child i  child mode (spawned by the parent)
//
// NOTE: ASCII only. nvcc's front end rejects non-ASCII inside string literals
//       on this machine (comments are fine, strings are not).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda.h>
#include <windows.h>

namespace {

constexpr std::size_t kGiB = 1024ull * 1024ull * 1024ull;
constexpr std::size_t kMainBytes  = 1536ull * 1024ull * 1024ull;   // 1.5 GB
constexpr std::size_t kChildBytes = 1536ull * 1024ull * 1024ull;   // 1.5 GB per child
constexpr std::size_t kReserve    = 4608ull * 1024ull * 1024ull;   // 4.5 GB of VA

void meminfo(const char* tag) {
    std::size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    std::printf("    [%s] free %.2f / total %.2f GiB\n", tag, f / (double)kGiB, t / (double)kGiB);
    std::fflush(stdout);
}

std::string exe_dir() {
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string s(buf);
    const std::size_t p = s.find_last_of("\\/");
    return p == std::string::npos ? std::string(".") : s.substr(0, p);
}

int run_child(int idx) {
    std::printf("[child %d] pid=%lu start\n", idx, GetCurrentProcessId());
    std::fflush(stdout);
    meminfo("child start");

    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, kChildBytes);
    std::printf("[child %d] cudaMalloc 1.5 GB -> %s\n", idx,
                e == cudaSuccess ? "OK" : cudaGetErrorString(e));
    std::fflush(stdout);
    if (e != cudaSuccess) { cudaGetLastError(); return 10; }

    cudaMemset(p, 0x11 + idx, 1 << 20);
    meminfo("child holding");

    cudaIpcMemHandle_t h{};
    const cudaError_t ei = cudaIpcGetMemHandle(&h, p);
    std::printf("[child %d] cudaIpcGetMemHandle -> %s\n", idx,
                ei == cudaSuccess ? "OK" : cudaGetErrorString(ei));
    std::fflush(stdout);

    char path[MAX_PATH];
    std::snprintf(path, sizeof(path), "%s\\ghost_child_%d.txt", exe_dir().c_str(), idx);
    FILE* f = std::fopen(path, "wb");
    if (f) {
        std::fprintf(f, "alloc=%d\nipc=%d\nbytes=%zu\n", (int)e, (int)ei, kChildBytes);
        if (ei == cudaSuccess) std::fwrite(&h, sizeof(h), 1, f);
        std::fclose(f);
    }

    char stop[MAX_PATH];
    std::snprintf(stop, sizeof(stop), "%s\\ghost_stop_%d.flag", exe_dir().c_str(), idx);
    for (int i = 0; i < 600; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        FILE* g = std::fopen(stop, "rb");
        if (g) { std::fclose(g); break; }
    }
    cudaFree(p);
    std::printf("[child %d] exit\n", idx);
    return 0;
}

void try_ipc_import() {
    std::printf("\n[E2] CUDA IPC (cross-process import)\n");
    char path[MAX_PATH];
    std::snprintf(path, sizeof(path), "%s\\ghost_child_0.txt", exe_dir().c_str());
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("    no child result file, skipped\n"); return; }
    int alloc = -1, ipc = -1; std::size_t bytes = 0;
    if (std::fscanf(f, "alloc=%d\nipc=%d\nbytes=%zu\n", &alloc, &ipc, &bytes) != 3) {
        std::fclose(f);
        std::printf("    bad result file, skipped\n");
        return;
    }
    cudaIpcMemHandle_t h{};
    if (ipc == 0) std::fread(&h, sizeof(h), 1, f);
    std::fclose(f);
    std::printf("    child cudaMalloc = %s, cudaIpcGetMemHandle = %s\n",
                alloc == 0 ? "OK" : "FAIL", ipc == 0 ? "OK" : "FAIL");
    if (ipc != 0) {
        std::printf("    --> export itself failed: on Windows the handle cannot be shared\n");
        return;
    }
    void* p = nullptr;
    const cudaError_t e = cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
    std::printf("    cudaIpcOpenMemHandle -> %s\n", e == cudaSuccess ? "OK" : cudaGetErrorString(e));
}

void try_vmm() {
    std::printf("\n[E3] VMM (contiguous VA + physical chunks)\n");
    CUdevice dev = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        std::printf("    cuInit/cuDeviceGet failed\n");
        return;
    }
    int vmm_ok = 0;
    cuDeviceGetAttribute(&vmm_ok, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev);
    std::printf("    VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = %d\n", vmm_ok);
    size_t gran = 0;
    cuMemGetAllocationGranularity(&gran, nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    std::printf("    min granularity = %.2f MiB\n", gran / (double)(1 << 20));

    CUcontext ctx = nullptr;
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);

    CUdeviceptr va = 0;
    CUresult r = cuMemAddressReserve(&va, kReserve, gran, 0, 0);
    std::printf("    cuMemAddressReserve(4.5 GiB) -> %d  va=0x%llX\n", (int)r,
                (unsigned long long)va);
    if (r != CUDA_SUCCESS) { std::printf("    --> VMM unavailable, E3 stops here\n"); return; }
    meminfo("after VA reserve");       // no physical yet: free should not move

    std::vector<CUmemGenericAllocationHandle> phys;
    const std::size_t chunk = 512ull * 1024ull * 1024ull;
    std::size_t mapped = 0;
    while (mapped + chunk <= kReserve) {
        CUmemGenericAllocationHandle h = 0;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        if (cuMemCreate(&h, chunk, &prop, 0) != CUDA_SUCCESS) {
            std::printf("    cuMemCreate chunk %zu failed (budget)\n", phys.size());
            cudaGetLastError();
            break;
        }
        if (cuMemMap(va + mapped, chunk, 0, h, 0) != CUDA_SUCCESS) {
            std::printf("    cuMemMap chunk %zu failed\n", phys.size());
            cuMemRelease(h);
            break;
        }
        CUmemAccessDesc ad{};
        ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ad.location.id = dev;
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuMemSetAccess(va + mapped, chunk, &ad, 1);
        phys.push_back(h);
        mapped += chunk;
        std::printf("    mapped chunk %zu (512 MiB) -> va+0x%llX\n", phys.size(),
                    (unsigned long long)mapped);
        if (phys.size() >= 3) break;
    }
    meminfo("after mapping physical");

    if (mapped == 0) { std::printf("    --> nothing mapped\n"); return; }

    std::printf("\n    read/write across %.2f GiB of contiguous VA\n", mapped / (double)kGiB);
    const cudaError_t rc = cudaMemset((void*)va, 0x5A, mapped);
    std::printf("    cudaMemset(va, 0x5A, %.2f GiB) -> %s\n", mapped / (double)kGiB,
                rc == cudaSuccess ? "OK" : cudaGetErrorString(rc));
    std::vector<unsigned char> host(1 << 20);
    int bad = 0;
    for (std::size_t off = 0; off < mapped; off += (1 << 20)) {
        if (cudaMemcpy(host.data(), (void*)(va + off), host.size(), cudaMemcpyDeviceToHost)
            != cudaSuccess) {
            ++bad; cudaGetLastError(); continue;
        }
        if (host[0] != 0x5A || host[host.size() - 1] != 0x5A) ++bad;
    }
    std::printf("    read-back mismatches: %d (0 = the whole range is really contiguous)\n", bad);

    for (std::size_t i = 0; i < phys.size(); ++i) {
        cuMemUnmap(va + i * chunk, chunk);
        cuMemRelease(phys[i]);
    }
    cuMemAddressFree(va, kReserve);
    cuDevicePrimaryCtxRelease(dev);
    std::printf("    VMM range released\n");
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0) {
        return run_child(std::atoi(argv[2]));
    }

    std::printf("=== FJEC_V2.4.0 ghost-process pool / VMM remap POC ===\n");
    meminfo("main start");

    std::printf("\n[E1] ghost processes: main holds 1.5 GB first\n");
    void* main_blk = nullptr;
    cudaError_t e = cudaMalloc(&main_blk, kMainBytes);
    std::printf("    main cudaMalloc 1.5 GB -> %s\n", e == cudaSuccess ? "OK" : cudaGetErrorString(e));
    if (e == cudaSuccess) cudaMemset(main_blk, 0x33, 1 << 20);
    meminfo("main holding 1.5 GB");

    const int NCHILD = 3;
    std::printf("\n    spawning %d children, 1.5 GB each\n", NCHILD);
    std::vector<PROCESS_INFORMATION> procs;
    for (int i = 0; i < NCHILD; ++i) {
        char cmd[MAX_PATH * 2];
        std::snprintf(cmd, sizeof(cmd), "\"%s\\ghost_pool.exe\" --child %d", exe_dir().c_str(), i);
        STARTUPINFOA si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        if (CreateProcessA(nullptr, cmd, nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            procs.push_back(pi);
            std::printf("    child %d -> pid=%lu\n", i, pi.dwProcessId);
        } else {
            std::printf("    child %d spawn failed (err=%lu)\n", i, GetLastError());
        }
    }

    for (int t = 0; t < 200; ++t) {
        int ready = 0;
        for (int i = 0; i < (int)procs.size(); ++i) {
            char p[MAX_PATH];
            std::snprintf(p, sizeof(p), "%s\\ghost_child_%d.txt", exe_dir().c_str(), i);
            FILE* f = std::fopen(p, "rb");
            if (f) { std::fclose(f); ++ready; }
        }
        if (ready == (int)procs.size()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    int child_ok = 0;
    for (int i = 0; i < (int)procs.size(); ++i) {
        char p[MAX_PATH];
        std::snprintf(p, sizeof(p), "%s\\ghost_child_%d.txt", exe_dir().c_str(), i);
        FILE* f = std::fopen(p, "rb");
        if (!f) { std::printf("    child %d: no result\n", i); continue; }
        int alloc = -1, ipc = -1; std::size_t bytes = 0;
        std::fscanf(f, "alloc=%d\nipc=%d\nbytes=%zu\n", &alloc, &ipc, &bytes);
        std::fclose(f);
        if (alloc == 0) ++child_ok;
        std::printf("    child %d: alloc=%s ipc=%s\n", i, alloc == 0 ? "OK" : "FAIL",
                    ipc == 0 ? "OK" : "FAIL");
    }
    meminfo("after all children hold");
    std::printf("    --> %d/%d children allocated 1.5 GB each\n", child_ok, (int)procs.size());

    try_ipc_import();
    try_vmm();

    for (int i = 0; i < (int)procs.size(); ++i) {
        char p[MAX_PATH];
        std::snprintf(p, sizeof(p), "%s\\ghost_stop_%d.flag", exe_dir().c_str(), i);
        FILE* f = std::fopen(p, "wb");
        if (f) { std::fprintf(f, "1"); std::fclose(f); }
    }
    for (auto& pi : procs) {
        WaitForSingleObject(pi.hProcess, 5000);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
    if (main_blk) cudaFree(main_blk);
    std::printf("\n=== POC done ===\n");
    return 0;
}
