// build/lt/d3d12_steal.cu - D3D12 graphics-budget escape, driver-API external memory POC
//
// Hypothesis: the WDDM budget CUDA gets (~2 GiB here) and the D3D12 graphics budget are
// separate books. If a D3D12 DEFAULT-heap buffer can take 4.5 GiB, and CUDA can import it
// as external memory, we get a device pointer charged to the graphics book -> 7B fits.
//
// Note on the API: CUDA 12.8 has NO cudaGraphicsD3D12RegisterResource (no
// cuda_d3d12_interop.h at all). The supported path is the driver API:
//     D3D12 committed resource (heap flag SHARED)
//        -> ID3D12Device::CreateSharedHandle
//        -> cuImportExternalMemory(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE)
//        -> cuExternalMemoryGetMappedBuffer   => CUdeviceptr
//
// Steps printed separately so a failure localises:
//   1. headless D3D12 device, print the DXGI graphics budget next to the CUDA free memory
//   2. allocate 1 / 2 / 3 / 4.5 GiB DEFAULT-heap buffers, keep the largest success
//   3. export a shared handle, import into CUDA, get the device pointer
//   4. cudaMemset + a read/write kernel over it, measure bandwidth
//
// usage: d3d12_steal.exe [GiB]
// ASCII only (nvcc front end rejects non-ASCII in string literals here).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cuda_runtime.h>
#include <cuda.h>

using Microsoft::WRL::ComPtr;

namespace {

constexpr std::size_t kGiB = 1024ull * 1024ull * 1024ull;

void cuda_info(const char* tag) {
    std::size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    std::printf("    [%s] cuda free %.2f / total %.2f GiB\n", tag, f / (double)kGiB, t / (double)kGiB);
    std::fflush(stdout);
}

__global__ void bw_kernel(float4* p, std::size_t n4, float4* sink) {
    const std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (std::size_t k = i; k < n4; k += stride) {
        float4 v = p[k];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        p[k] = v;
    }
    if (acc.x == 1e30f) *sink = acc;
}

const char* cu_str(CUresult r) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    return s;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const double want_gib = (argc > 1) ? std::atof(argv[1]) : 4.5;

    std::printf("=== D3D12 graphics-budget escape POC (wanted %.2f GiB) ===\n", want_gib);
    cuda_info("start");

    // ---- 1. headless device + both budgets ----
    ComPtr<ID3D12Device> dev;
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    std::printf("\n[1] D3D12CreateDevice(null, 12_0) -> hr=0x%08lX %s\n", (unsigned long)hr,
                SUCCEEDED(hr) ? "OK" : "FAIL");
    if (FAILED(hr)) return 1;

    {
        ComPtr<IDXGIFactory4> fac;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&fac)))) {
            ComPtr<IDXGIAdapter1> ad;
            if (SUCCEEDED(fac->EnumAdapters1(0, &ad))) {
                DXGI_ADAPTER_DESC1 d{};
                ad->GetDesc1(&d);
                std::printf("    adapter: %ls  dedicated %.2f GiB\n", d.Description,
                            d.DedicatedVideoMemory / (double)kGiB);
                ComPtr<IDXGIAdapter3> ad3;
                if (SUCCEEDED(ad.As(&ad3))) {
                    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
                    if (SUCCEEDED(ad3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
                        std::printf("    DXGI LOCAL budget %.2f GiB | usage %.2f GiB | available %.2f GiB\n",
                                    info.Budget / (double)kGiB, info.CurrentUsage / (double)kGiB,
                                    (info.Budget - info.CurrentUsage) / (double)kGiB);
                    }
                    if (SUCCEEDED(ad3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                        std::printf("    DXGI NON-LOCAL budget %.2f GiB | available %.2f GiB\n",
                                    info.Budget / (double)kGiB,
                                    (info.Budget - info.CurrentUsage) / (double)kGiB);
                    }
                }
            }
        }
    }

    // ---- 2. allocate on a SHARED default heap, keep the largest success ----
    std::printf("\n[2] D3D12 DEFAULT-heap SHARED buffers\n");
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    hp.CreationNodeMask = 1;
    hp.VisibleNodeMask = 1;

    ComPtr<ID3D12Resource> big;
    std::size_t big_bytes = 0;
    const double tries[] = {want_gib};   // only the requested size: 2+ GiB resets the device
    std::vector<double> tried;
    for (double g : tries) {
        bool dup = false;
        for (double t : tried) if (t == g) dup = true;
        if (dup) continue;
        tried.push_back(g);

        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = (UINT64)(g * (double)kGiB);
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        

        ComPtr<ID3D12Resource> r;
        const HRESULT h2 = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&r));
        std::printf("    %.2f GiB -> hr=0x%08lX %s\n", g, (unsigned long)h2,
                    SUCCEEDED(h2) ? "OK" : "FAIL");
        if (SUCCEEDED(h2)) { big = r; big_bytes = (std::size_t)rd.Width; }
    }
    if (!big) { std::printf("    --> no buffer at all, hypothesis dead at step 2\n"); return 2; }
    std::printf("    kept the largest success: %.2f GiB\n", big_bytes / (double)kGiB);

    // ---- 3. shared handle -> cuImportExternalMemory ----
    std::printf("\n[3] D3D12 shared handle -> CUDA external memory\n");
    HANDLE shared = nullptr;
    hr = dev->CreateSharedHandle(big.Get(), nullptr, GENERIC_ALL, nullptr, &shared);
    std::printf("    CreateSharedHandle -> hr=0x%08lX handle=%p\n", (unsigned long)hr, shared);
    if (FAILED(hr)) return 3;

    if (cuInit(0) != CUDA_SUCCESS) { std::printf("    cuInit failed\n"); return 3; }
    CUdevice cudev = 0;
    cuDeviceGet(&cudev, 0);
    CUcontext ctx = nullptr;
    cuDevicePrimaryCtxRetain(&ctx, cudev);
    cuCtxSetCurrent(ctx);

    CUexternalMemory ext{};
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC ed{};
    ed.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    ed.handle.win32.handle = (void*)shared;

    CUresult r = cuImportExternalMemory(&ext, &ed);
    std::printf("    cuImportExternalMemory -> %d (%s)\n", (int)r, cu_str(r));
    if (r != CUDA_SUCCESS) return 3;

    CUdeviceptr dptr = 0;
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
    bd.offset = 0;
    bd.size = 0;
    r = cuExternalMemoryGetMappedBuffer(&dptr, ext, &bd);
    std::printf("    cuExternalMemoryGetMappedBuffer -> %d (%s) ptr=0x%llX\n", (int)r, cu_str(r),
                (unsigned long long)dptr);
    if (r != CUDA_SUCCESS) return 3;

    // ---- 4. use it ----
    std::printf("\n[4] CUDA over the D3D12 memory\n");
    void* ptr = (void*)dptr;
    const std::size_t mapped = big_bytes;
    const cudaError_t em = cudaMemset(ptr, 0x5A, mapped);
    std::printf("    cudaMemset(%.2f GiB) -> %s\n", mapped / (double)kGiB,
                em == cudaSuccess ? "OK" : cudaGetErrorString(em));
    cudaGetLastError();

    std::vector<unsigned char> probe(64);
    int bad = 0, spots = 0;
    for (std::size_t off = 0; off + probe.size() <= mapped; off += mapped / 8, ++spots) {
        if (cudaMemcpy(probe.data(), (char*)ptr + off, probe.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
            ++bad; cudaGetLastError(); continue;
        }
        if (probe[0] != 0x5A) ++bad;
    }
    std::printf("    spot reads: %d bad of %d\n", bad, spots);

    float4* sink = nullptr;
    cudaMalloc(&sink, sizeof(float4));
    const std::size_t n4 = mapped / sizeof(float4);
    const int threads = 256, blocks = 1024;
    bw_kernel<<<blocks, threads>>>((float4*)ptr, n4, sink);
    cudaDeviceSynchronize();

    const int reps = 3;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) bw_kernel<<<blocks, threads>>>((float4*)ptr, n4, sink);
    const cudaError_t ek = cudaDeviceSynchronize();
    const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    std::printf("    kernel x%d -> %s  %.2f ms total (%.2f ms/iter)\n", reps,
                ek == cudaSuccess ? "OK" : cudaGetErrorString(ek), ms, ms / reps);
    if (ek == cudaSuccess) {
        std::printf("    bandwidth (read+write) = %.1f GB/s\n",
                    (double)mapped * 2.0 * reps / (ms * 1e-3) / 1e9);
    }
    cudaGetLastError();

    cuMemFree(dptr);
    cuDestroyExternalMemory(ext);
    cuDevicePrimaryCtxRelease(cudev);
    if (shared) CloseHandle(shared);
    cudaFree(sink);
    std::printf("\n=== POC done ===\n");
    return 0;
}
