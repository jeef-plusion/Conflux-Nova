// nova_v2/code/device/sycl/device.cpp — 设备层实现
#include "device.h"

#include <cinttypes>
#include <cstring>
#include <new>
#include <stdexcept>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace nova {
namespace syclb {
namespace {

const char* kTagName[] = {"weights", "kv", "act", "scratch", "io"};
static_assert(static_cast<int>(MemTag::Count) == 5, "update kTagName");

std::string bytes_str(std::uint64_t b) {
    char buf[64];
    const double mib = static_cast<double>(b) / (1024.0 * 1024.0);
    std::snprintf(buf, sizeof(buf), "%" PRIu64 " B (%.1f MiB)", b, mib);
    return std::string(buf);
}

[[noreturn]] void oom(const char* what, MemTag tag, std::uint64_t bytes,
                      std::uint64_t reserved, std::uint64_t dev_total,
                      const char* api) {
    const HostMem hm = host_mem_snapshot();
    std::string msg;
    char line[512];
    std::snprintf(line, sizeof(line),
                  "SYCL 分配失败 (%s): 申请 %s, tag=%s\n"
                  "        本进程已占用 %" PRIu64 " MiB\n"
                  "        设备 global_mem_size = %" PRIu64 " MiB\n"
                  "        系统物理内存 total = %" PRIu64 " MiB, avail = %" PRIu64 " MiB\n"
                  "        单次分配上限 %" PRIu64 " MiB, 单块权重上限 %" PRIu64 " MiB\n"
                  "        处置：不要重试、不要降级 —— 按 docs/SYCL_PORT.md 缩小 KV/上下文或换更小的量化模型。",
                  api, bytes_str(bytes).c_str(), kTagName[static_cast<int>(tag)],
                  reserved / (1024ull * 1024ull),
                  dev_total / (1024ull * 1024ull),
                  hm.total / (1024ull * 1024ull), hm.avail / (1024ull * 1024ull),
                  profile::kMaxAllocBytes / (1024ull * 1024ull),
                  profile::kMaxChunkBytes / (1024ull * 1024ull));
    msg = line;
    (void)what;
    NOVA_ABORT(msg.c_str());
}

} // namespace

const char* mem_tag_name(MemTag t) { return kTagName[static_cast<int>(t)]; }

const char* dp4a_mode_name(int mode) {
    switch (mode) {
        case 0: return "scalar (no hw intrinsic)";
        case 1: return "sycl::dp4a(...,int)";
        case 2: return "sycl::dp4a(...,vec<int32_t,1>)";
        case 3: return "manual 4xMAC (dpct-style)";
        default: return "?";
    }
}

HostMem host_mem_snapshot() {
    HostMem hm{};
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        hm.total = static_cast<std::uint64_t>(ms.ullTotalPhys);
        hm.avail = static_cast<std::uint64_t>(ms.ullAvailPhys);
    }
#endif
    return hm;
}

// ============================================================================
// DeviceCaps::report
// ============================================================================
std::string DeviceCaps::report() const {
    std::string s;
    char line[512];
    auto add = [&](const char* fmt, auto... args) {
        std::snprintf(line, sizeof(line), fmt, args...);
        s += line;
        s += '\n';
    };
    add("  device        : %s", name.c_str());
    add("  vendor/driver : %s / %s", vendor.c_str(), driver.c_str());
    add("  runtime       : %s", version.c_str());
    add("  Xe cores(EU)  : %d   clock = %d MHz", compute_units, clock_mhz);
    add("  global mem    : %" PRIu64 " MiB", global_mem_bytes / (1024ull * 1024ull));
    add("  max alloc     : %" PRIu64 " MiB (profile 假设 %" PRIu64 " MiB)", max_alloc_bytes / (1024ull * 1024ull),
        profile::kMaxAllocBytes / (1024ull * 1024ull));
    add("  local mem     : %" PRIu64 " KiB (profile 假设 %d KiB)", local_mem_bytes / 1024ull,
        profile::kLocalMemBytes / 1024);
    add("  max workgroup : %zu (profile 假设 %d)", max_wg_size, profile::kMaxWorkGroupSize);
    char sgs[128] = {0};
    for (int i = 0; i < sg_size_count; ++i) {
        char t[16];
        std::snprintf(t, sizeof(t), "%zu ", sg_sizes[i]);
        std::strcat(sgs, t);
    }
    add("  subgroup      : %zu  (sizes: %s) 使用 %d", sg_size, sgs, kSyclSg);
    add("  USM device    : %s ; shared : %s", usm_device ? "yes" : "NO", usm_shared ? "yes" : "NO");
    add("  fp16          : %s", fp16 ? "yes" : "no");
    add("  dp4a 路径     : %s", dp4a_mode_name(dp4a_mode));
    return s;
}

// ============================================================================
// Device
// ============================================================================
Device::Device() {
    std::vector<sycl::device> gpus;
    try {
        gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    } catch (const std::exception& e) {
        std::string m = std::string("枚举 GPU 失败: ") + e.what();
        NOVA_ABORT(m.c_str());
    }
    if (gpus.empty()) {
        NOVA_ABORT("没有任何 SYCL GPU 设备。装 Intel oneAPI 运行时 + 显卡驱动后重试；"
                   "本后端不做 CPU 回退（docs/SYCL_PORT.md §7）。");
    }

    // 只有一个 iGPU 时装哪一个都一样；多 GPU 时优先名字里带 Arc 的那块。
    std::size_t pick = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        const std::string n = gpus[i].get_info<sycl::info::device::name>();
        if (n.find("Arc") != std::string::npos) { pick = i; break; }
    }
    dev_ = gpus[pick];

    caps_.name    = dev_.get_info<sycl::info::device::name>();
    caps_.vendor  = dev_.get_info<sycl::info::device::vendor>();
    caps_.driver  = dev_.get_info<sycl::info::device::driver_version>();
    caps_.version = dev_.get_info<sycl::info::device::version>();
    caps_.global_mem_bytes = dev_.get_info<sycl::info::device::global_mem_size>();
    caps_.local_mem_bytes  = dev_.get_info<sycl::info::device::local_mem_size>();
    caps_.max_alloc_bytes  = dev_.get_info<sycl::info::device::max_mem_alloc_size>();
    caps_.max_wg_size      = dev_.get_info<sycl::info::device::max_work_group_size>();
    caps_.compute_units    = static_cast<int>(dev_.get_info<sycl::info::device::max_compute_units>());
    caps_.clock_mhz        = static_cast<int>(dev_.get_info<sycl::info::device::max_clock_frequency>());
    caps_.sg_size          = dev_.get_info<sycl::info::device::sub_group_sizes>().empty()
                                 ? 0
                                 : dev_.get_info<sycl::info::device::sub_group_sizes>().front();
    {
        const auto sizes = dev_.get_info<sycl::info::device::sub_group_sizes>();
        caps_.sg_size_count = static_cast<int>(sizes.size() < 8 ? sizes.size() : 8);
        for (int i = 0; i < caps_.sg_size_count; ++i) caps_.sg_sizes[i] = sizes[static_cast<std::size_t>(i)];
        // 我们请求的是 kSyclSg，不是"默认值"；只要它出现在支持列表里就成立。
        bool found = false;
        for (std::size_t i = 0; i < sizes.size(); ++i) found = found || (sizes[i] == static_cast<std::size_t>(kSyclSg));
        if (!found) {
            std::string m = "设备不支持 subgroup=" + std::to_string(kSyclSg) +
                            "（kernel 用 reqd_sub_group_size 写死）。改 -DNOVA_SYCL_SG=16 重编。";
            NOVA_ABORT(m.c_str());
        }
    }
    caps_.usm_device = dev_.has(sycl::aspect::usm_device_allocations);
    caps_.usm_shared = dev_.has(sycl::aspect::usm_shared_allocations);
    caps_.fp16       = dev_.has(sycl::aspect::fp16);
    caps_.dp4a_mode  = kNovaDp4aMode;

    if (!caps_.usm_device) {
        NOVA_ABORT("设备不支持 USM device allocation（aspect::usm_device_allocations=false）。"
                   "本后端的全部 buffer 依赖 USM，没有 buffer/accessor 回退路径。");
    }
    if (caps_.local_mem_bytes < static_cast<std::uint64_t>(profile::kLocalMemBytes)) {
        std::printf("[nova][sycl][WARN] local_mem_size = %" PRIu64
                    " KiB < profile 的 %d KiB；attn prefill 的 KV tile 需要 16 KiB，仍可运行。\n",
                    caps_.local_mem_bytes / 1024ull, profile::kLocalMemBytes / 1024);
    }
    if (caps_.global_mem_bytes < profile::kDeviceBudget / 2) {
        std::printf("[nova][sycl][WARN] global_mem_size 只有 %" PRIu64
                    " MiB，小于 profile 的 8 GiB 预算的一半；7B 权重放不下（会 OOM 并打印明细）。\n",
                    caps_.global_mem_bytes / (1024ull * 1024ull));
    }

    ctx_ = sycl::context(dev_);
    // 单流顺序队列：与 CUDA 的 stream 语义一一对应，算子之间不需要显式 event 依赖。
    q_ = sycl::queue(ctx_, dev_, sycl::property::queue::in_order{});
}

Device::~Device() {
    if (reserved_ != 0) {
        // 单次运行的程序（基准/自检）走到进程结束属正常；库形态下应该全为 0。
        std::printf("[nova][sycl] 进程退出时仍持有 %" PRIu64 " MiB USM（重置/析构前未归还）\n",
                    reserved_ / (1024ull * 1024ull));
    }
}

void* Device::raw_alloc(std::uint64_t bytes, MemTag tag, const char* what, bool shared) {
    if (bytes == 0) return nullptr;
    if (bytes > profile::kMaxAllocBytes) {
        // 单次分配上限是驱动的硬限制；宁可在这里说清楚，也不要让驱动返回 nullptr。
        std::string m = "单次分配 " + bytes_str(bytes) + " 超过 profile 上限 " +
                        bytes_str(profile::kMaxAllocBytes) + "（tag=" + kTagName[static_cast<int>(tag)] +
                        "）。大的权重必须走 ChunkedWeight 分块。";
        NOVA_ABORT(m.c_str());
    }
    void* p = nullptr;
    try {
        // 注意：malloc_* 是 template<typename T>(size_t count, ...)，T 不能从 size 推导，
        // 必须显式给 <std::uint8_t>（此时 count 就是字节数）。
        p = shared ? static_cast<void*>(sycl::malloc_shared<std::uint8_t>(bytes, q_))
                   : static_cast<void*>(sycl::malloc_device<std::uint8_t>(bytes, q_));
    } catch (const sycl::exception& e) {
        oom(what, tag, bytes, reserved_, caps_.global_mem_bytes, shared ? "malloc_shared" : "malloc_device");
    } catch (const std::bad_alloc&) {
        oom(what, tag, bytes, reserved_, caps_.global_mem_bytes, shared ? "malloc_shared" : "malloc_device");
    }
    if (p == nullptr) oom(what, tag, bytes, reserved_, caps_.global_mem_bytes, shared ? "malloc_shared" : "malloc_device");
    note_alloc(bytes, tag);
    return p;
}

void* Device::alloc(std::uint64_t bytes, MemTag tag, const char* what) {
    return raw_alloc(bytes, tag, what, false);
}
void* Device::alloc_shared(std::uint64_t bytes, MemTag tag, const char* what) {
    return raw_alloc(bytes, tag, what, true);
}

void Device::release(void* p, MemTag tag, std::uint64_t bytes) {
    if (p == nullptr) return;
    sycl::free(p, ctx_);
    note_free(bytes, tag);
}

void Device::note_alloc(std::uint64_t bytes, MemTag tag) {
    reserved_ += bytes;
    if (reserved_ > peak_) peak_ = reserved_;
    MemStat& s = stats_[static_cast<int>(tag)];
    s.live += bytes;
    s.allocs += 1;
    if (reserved_ > profile::kDeviceBudget) {
        char line[256];
        std::snprintf(line, sizeof(line),
                      "已保留 %.1f MiB > 8 GiB 设备预算；继续下去会撞 Windows 的 WDDM 限制。",
                      static_cast<double>(reserved_) / (1024.0 * 1024.0));
        NOVA_ABORT(line);
    }
}

void Device::note_free(std::uint64_t bytes, MemTag tag) {
    reserved_ -= bytes;
    stats_[static_cast<int>(tag)].live -= bytes;
}

void Device::upload(void* dst, const void* src, std::uint64_t bytes) {
    if (bytes == 0) return;
    q_.memcpy(dst, src, bytes);
    q_.wait_and_throw();
}

void Device::download(void* dst, const void* src, std::uint64_t bytes) {
    if (bytes == 0) return;
    q_.memcpy(dst, src, bytes);
    q_.wait_and_throw();
}

void Device::zero(void* dst, std::uint64_t bytes) {
    if (bytes == 0) return;
    q_.memset(dst, 0, bytes);
}

void Device::sync() { q_.wait_and_throw(); }

void Device::print_memory_report() const {
    std::printf("[nova][sycl] USM 记账: live=%.1f MiB  peak=%.1f MiB\n",
                static_cast<double>(reserved_) / (1024.0 * 1024.0),
                static_cast<double>(peak_) / (1024.0 * 1024.0));
    for (int i = 0; i < static_cast<int>(MemTag::Count); ++i) {
        if (stats_[i].allocs == 0) continue;
        std::printf("            %-8s live=%8.1f MiB  peak=%8.1f MiB  allocs=%" PRIu64 "\n",
                    kTagName[i],
                    static_cast<double>(stats_[i].live) / (1024.0 * 1024.0),
                    static_cast<double>(stats_[i].peak) / (1024.0 * 1024.0),
                    stats_[i].allocs);
    }
}

void Device::print_banner() {
    std::printf("[nova][sycl] 目标 profile（写死，不探测）:\n");
    std::printf("            Xe cores=%d  subgroup=%d  local=%d KiB  wg<=%d\n",
                profile::kXeCores, kSyclSg, profile::kLocalMemBytes / 1024, profile::kMaxWorkGroupSize);
    std::printf("            单次分配<=%" PRIu64 " B  单块权重<=%" PRIu64 " B  设备预算=%" PRIu64 " MiB\n",
                profile::kMaxAllocBytes, profile::kMaxChunkBytes, profile::kDeviceBudget / (1024ull * 1024ull));
    std::printf("            设计带宽=%.1f GB/s（理论 %.1f）dp4a=%s\n",
                profile::kDesignBandwidthGBs, profile::kTheoreticalBandwidthGBs,
                profile::kDp4a ? "hardware" : "off");
}

// ============================================================================
// ChunkedWeight
// ============================================================================
void ChunkedWeight::create(Device& dev, std::int64_t rows, std::int64_t row_bytes, const char* what) {
    rows_      = rows;
    row_bytes_ = row_bytes;
    chunks_.clear();
    if (rows <= 0 || row_bytes <= 0) return;

    // 每块最多 kMaxChunkBytes，向下取整到整行；保证每块至少一行。
    std::int64_t rows_per_chunk = static_cast<std::int64_t>(profile::kMaxChunkBytes / static_cast<std::uint64_t>(row_bytes));
    if (rows_per_chunk < 1) rows_per_chunk = 1;

    std::int64_t r = 0;
    while (r < rows) {
        const std::int64_t r1 = (r + rows_per_chunk < rows) ? (r + rows_per_chunk) : rows;
        Chunk c;
        c.row0  = r;
        c.row1  = r1;
        c.bytes = static_cast<std::uint64_t>(r1 - r) * static_cast<std::uint64_t>(row_bytes);
        c.ptr   = static_cast<std::uint8_t*>(dev.alloc(c.bytes, MemTag::Weights, what));
        chunks_.push_back(c);
        bytes_ += c.bytes;
        r = r1;
    }
}

void ChunkedWeight::upload(Device& dev, const void* src) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(src);
    for (std::size_t i = 0; i < chunks_.size(); ++i) {
        dev.upload(chunks_[i].ptr, p, chunks_[i].bytes);
        p += chunks_[i].bytes;
    }
}

void ChunkedWeight::upload_chunk(Device& dev, int i, const void* src) {
    const Chunk& c = chunks_[static_cast<std::size_t>(i)];
    dev.upload(c.ptr, src, c.bytes);
}

void ChunkedWeight::destroy(Device& dev) {
    for (std::size_t i = 0; i < chunks_.size(); ++i) {
        dev.release(chunks_[i].ptr, MemTag::Weights, chunks_[i].bytes);
    }
    chunks_.clear();
    bytes_ = 0;
}

void ChunkedWeight::locate(std::int64_t r, int& chunk_idx, std::int64_t& row_in_chunk) const {
    for (std::size_t i = 0; i < chunks_.size(); ++i) {
        if (r >= chunks_[i].row0 && r < chunks_[i].row1) {
            chunk_idx    = static_cast<int>(i);
            row_in_chunk = r - chunks_[i].row0;
            return;
        }
    }
    chunk_idx    = -1;
    row_in_chunk = -1;
}

} // namespace syclb
} // namespace nova
