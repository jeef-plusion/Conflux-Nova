// nova_v2/code/device/sycl/device.h — SYCL 后端设备层
// 目标：Intel Arc 140T (Core Ultra 7 255H / Arrow Lake-H) / Win11 / oneAPI DPC++。
//
// 本文件只做四件事（算子层不碰 sycl::queue / 不做分配）：
//   1. 把目标硬件 profile 写死成编译期常量（不做能力探测、不做降级分支）；
//   2. USM 分配 / 上传 / 下载 / 同步，OOM 立刻报错并打印 free/total；
//   3. 分块权重描述符（每块 ≤ 1 GiB，单次分配 ≤ 3.999 GiB）；
//   4. 与 CUDA 版 ActQ8 / Q4KWeight 一一对应的 POD 视图（不含任何 CUDA 头）。
//
// 铁律（沿用 include/nova/config.h 的口径）：算子内零分配；分配失败 = abort。
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nova/config.h"

namespace nova {
namespace syclb {

// ============================================================================
// 1. 目标硬件 profile —— 写死。数值来自 docs/BACKEND_INTEL_255H.md 的实测/推算。
//    这些常量是唯一真源；init 时只"打印"运行期实测值用于核对，绝不用它改行为。
// ============================================================================
namespace profile {

inline constexpr int kXeCores          = 8;      // Arc 140T: 8 Xe core / 1024 fp32 lane
inline constexpr int kSubgroupSize     = 32;     // 驱动默认 32；可用 -DNOVA_SYCL_SG=16 切
inline constexpr int kLocalMemBytes    = 32 * 1024;   // Windows 驱动 maxComputeSharedMemorySize
inline constexpr int kMaxWorkGroupSize = 1024;

inline constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kMaxAllocBytes = 4294901760ull;   // 3.999 GiB：单次分配上限
inline constexpr std::uint64_t kMaxChunkBytes = 1ull * kGiB;     // 1.0 GiB：单 buffer 上限
inline constexpr std::uint64_t kDeviceBudget  = 8ull * kGiB;     // 32 GB 系统内存里划给 iGPU 的额度

inline constexpr double kDesignBandwidthGBs      = 100.0;   // 性能模型取值（理论 136.5 的 ~73%）
inline constexpr double kTheoreticalBandwidthGBs = 136.5;   // LPDDR5x-8533 / 128 bit

inline constexpr bool kDp4a = true;      // dotPacked4x8EXT / sycl::dp4a 直通
inline constexpr bool kFp16 = true;      // shaderFloat16

} // namespace profile

// ---------------------------------------------------------------------------
// 编译期开关（build_sycl.bat 会按探针结果注入 NOVA_SYCL_DP4A）
//   NOVA_SYCL_SG   : 32（默认）/ 16。所有 kernel 用 [[sycl::reqd_sub_group_size]] 写死。
//   NOVA_SYCL_DP4A : 0 = 纯标量（可强制用于对拍）
//                    1 = sycl::dp4a(vec<int8_t,4>, vec<int8_t,4>, int)
//                    2 = sycl::dp4a(vec<int8_t,4>, vec<int8_t,4>, vec<int32_t,1>)
//                    3 = 手写 4 乘加（dpct::dp4a 等价式，编译器可自行识别）
//   探针不通过时 build_sycl.bat 落回 0 并在基准里打印大字告警，不静默。
// ---------------------------------------------------------------------------
#ifndef NOVA_SYCL_SG
#  define NOVA_SYCL_SG 32
#endif
inline constexpr int kSyclSg = NOVA_SYCL_SG;
static_assert(kSyclSg == 16 || kSyclSg == 32, "subgroup 只支持 16 / 32");

#ifndef NOVA_SYCL_DP4A
#  define NOVA_SYCL_DP4A 0
#endif
inline constexpr int kNovaDp4aMode = NOVA_SYCL_DP4A;

// 0/3 是软件路径；1/2 是硬件 dp4a 直通
inline constexpr bool kDp4aHardware = (kNovaDp4aMode == 1 || kNovaDp4aMode == 2);
const char* dp4a_mode_name(int mode);

// ============================================================================
// 2. 运行期实测胶囊（只用于打印与断言，不参与任何分支决策）
// ============================================================================
struct DeviceCaps {
    std::string name;
    std::string vendor;
    std::string driver;
    std::string version;          // sycl 后端版本串
    std::uint64_t global_mem_bytes = 0;
    std::uint64_t local_mem_bytes  = 0;
    std::uint64_t max_alloc_bytes  = 0;   // info::device::max_mem_alloc_size
    std::size_t   max_wg_size      = 0;
    std::size_t   sg_size          = 0;
    std::size_t   sg_sizes[8]      = {};
    int           sg_size_count    = 0;
    int           compute_units    = 0;
    int           clock_mhz        = 0;
    bool          usm_device       = false;
    bool          usm_shared       = false;
    bool          fp16             = false;
    int           dp4a_mode        = -1;   // 见上方 NOVA_SYCL_DP4A

    std::string report() const;
};

// ============================================================================
// 3. 内存记账
// ============================================================================
enum class MemTag : int { Weights = 0, Kv, Act, Scratch, Io, Count };
const char* mem_tag_name(MemTag t);

struct MemStat {
    std::uint64_t live  = 0;
    std::uint64_t peak  = 0;
    std::uint64_t allocs = 0;
};

// 系统内存快照（Windows: GlobalMemoryStatusEx；其它平台返回 0）
struct HostMem {
    std::uint64_t total = 0;
    std::uint64_t avail = 0;
};
HostMem host_mem_snapshot();

// ============================================================================
// 4. 设备 + 队列 + USM
// ============================================================================
class Device {
public:
    Device();
    ~Device();

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;

    sycl::queue&       q()          { return q_; }
    const sycl::queue& q() const    { return q_; }
    const DeviceCaps&  caps() const { return caps_; }

    // 分配。bytes == 0 返回 nullptr。失败立即打印 free/total 并 abort。
    void* alloc(std::uint64_t bytes, MemTag tag, const char* what);
    void* alloc_shared(std::uint64_t bytes, MemTag tag, const char* what);
    void  release(void* p, MemTag tag, std::uint64_t bytes);

    void  upload(void* dst, const void* src, std::uint64_t bytes);
    void  download(void* dst, const void* src, std::uint64_t bytes);
    void  zero(void* dst, std::uint64_t bytes);
    void  sync();

    std::uint64_t reserved() const { return reserved_; }
    std::uint64_t peak()     const { return peak_; }

    void print_memory_report() const;
    static void print_banner();

private:
    void* raw_alloc(std::uint64_t bytes, MemTag tag, const char* what, bool shared);
    void  note_alloc(std::uint64_t bytes, MemTag tag);
    void  note_free(std::uint64_t bytes, MemTag tag);

    sycl::device  dev_;
    sycl::context ctx_;
    sycl::queue   q_;
    DeviceCaps    caps_{};

    std::uint64_t reserved_ = 0;
    std::uint64_t peak_     = 0;
    MemStat       stats_[static_cast<int>(MemTag::Count)]{};
};

// ============================================================================
// 5. 分块权重描述符
//    Windows 上单个 storage buffer 上限 1.0 GiB，7B 的 3.99 GiB 权重必须切片。
//    按"整行"切（一行 = cols/256 * 144 B ≤ 2016 B，远小于 1 GiB），保证任何一层
//    的任意一行都完整落在某一个块里，kernel 里没有跨块指针运算。
// ============================================================================
class ChunkedWeight {
public:
    struct Chunk {
        std::uint8_t* ptr  = nullptr;
        std::int64_t  row0 = 0;      // 闭区间起点
        std::int64_t  row1 = 0;      // 开区间终点
        std::uint64_t bytes = 0;
    };

    // rows 行、每行 row_bytes 字节；按 ≤ profile::kMaxChunkBytes 切块。
    void create(Device& dev, std::int64_t rows, std::int64_t row_bytes, const char* what);
    void upload(Device& dev, const void* src);         // 按块顺序上传，src 为整块张量
    void upload_chunk(Device& dev, int i, const void* src);
    void destroy(Device& dev);

    std::int64_t rows()      const { return rows_; }
    std::int64_t row_bytes() const { return row_bytes_; }
    std::uint64_t bytes()    const { return bytes_; }
    int          n_chunks()  const { return static_cast<int>(chunks_.size()); }
    const Chunk& chunk(int i) const { return chunks_[static_cast<std::size_t>(i)]; }
    const std::uint8_t* base() const { return chunks_.empty() ? nullptr : chunks_[0].ptr; }

    // 第 r 行所在块与块内行号（调试/对拍用）
    void locate(std::int64_t r, int& chunk_idx, std::int64_t& row_in_chunk) const;

private:
    std::vector<Chunk> chunks_;
    std::int64_t  rows_      = 0;
    std::int64_t  row_bytes_ = 0;
    std::uint64_t bytes_     = 0;
};

// ============================================================================
// 6. 算子层用的 POD 视图（与 include/nova/kernel/types.h 语义逐字段一致，
//    只是不含 cuda_runtime.h。kernel 里按值传参。）
// ============================================================================
struct Q4KWeightView {          // 单块视图（launcher 按块循环，kernel 只见一块）
    const std::uint8_t* blocks = nullptr;   // rows * (cols/256) 个 block_q4_K，行主序
    std::int64_t        rows   = 0;
    std::int64_t        cols   = 0;
};

struct ActQ8View {              // 每 32 元素一个 fp32 scale，每 16 元素一个 int32 和
    std::int8_t* qs = nullptr;  // rows * cols
    float*       d  = nullptr;  // rows * (cols/32)
    std::int32_t* s = nullptr;  // rows * (cols/16)
};

// Q4_K 块几何（与 include/nova/quant/q4_k.h 同源）
inline constexpr int kQK        = nova::kQK;          // 256
inline constexpr int kBlockBytes = nova::kBlockBytes; // 144

inline std::int64_t q4k_row_bytes(std::int64_t cols) { return (cols / kQK) * kBlockBytes; }

inline std::uint64_t act_q8_bytes(std::int64_t rows, std::int64_t cols) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
}
inline std::uint64_t act_d_bytes(std::int64_t rows, std::int64_t cols) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols / 32) * 4ull;
}
inline std::uint64_t act_s_bytes(std::int64_t rows, std::int64_t cols) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols / 16) * 4ull;
}

} // namespace syclb
} // namespace nova
