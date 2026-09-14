// alloc.h — 对齐内存分配工具
// =========================================================================
// 提供:
//   1. AlignedBuffer  — RAII 对齐缓冲区，可复用，自动释放
//   2. aligned_alloc  — 对齐分配（封装，统一接口）
//   3. thread_local 缓冲区管理
//
// 所有 SIMD 缓冲区应使用此文件中的分配函数，确保对齐。

#ifndef ALLOC_H_
#define ALLOC_H_

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>

// 默认对齐字节数（AVX2=32, AVX-512=64）
#ifndef SIMD_ALIGN
#define SIMD_ALIGN 64
#endif

// ===========================================================================
// 对齐分配/释放（封装标准函数）
// ===========================================================================

static inline void* aligned_malloc(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, SIMD_ALIGN);
#else
    // aligned_alloc 要求大小是对齐的倍数
    size_t aligned_bytes = (bytes + SIMD_ALIGN - 1) & ~(SIMD_ALIGN - 1);
    return std::aligned_alloc(SIMD_ALIGN, aligned_bytes);
#endif
}

static inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// 分配 n 个 float 的对齐缓冲区
static inline float* aligned_alloc_float(size_t n) {
    return (float*)aligned_malloc(n * sizeof(float));
}


#endif // ALLOC_H_


// ===========================================================================
// AlignedBuffer — RAII 对齐缓冲区
// ===========================================================================
// 用途：需要频繁使用但大小固定的临时缓冲区
// 特性：
//   - 构造时分配，析构时释放（RAII）
//   - 可复用：resize() 只在需要更大时才重新分配
//   - 禁止拷贝，允许移动
//   - 热路径中 .data() 返回裸指针，零开销

struct AlignedBuffer {
    float* ptr_ = nullptr;
    size_t cap_ = 0;  // 容量（float 数量）

    AlignedBuffer() = default;
    
    explicit AlignedBuffer(size_t n) {
        resize(n);
    }

    ~AlignedBuffer() {
        aligned_free(ptr_);
    }

    // 禁止拷贝
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // 允许移动
    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr_(o.ptr_), cap_(o.cap_) {
        o.ptr_ = nullptr;
        o.cap_ = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            aligned_free(ptr_);
            ptr_ = o.ptr_;
            cap_ = o.cap_;
            o.ptr_ = nullptr;
            o.cap_ = 0;
        }
        return *this;
    }

    // 确保容量至少为 n（只增不减）
    void resize(size_t n) {
        if (n > cap_) {
            aligned_free(ptr_);
            ptr_ = aligned_alloc_float(n);
            cap_ = n;
        }
    }

    // 清零内容（不释放内存）
    void zero() {
        if (ptr_) std::memset(ptr_, 0, cap_ * sizeof(float));
    }

    // 访问
    float* data() { return ptr_; }
    const float* data() const { return ptr_; }
    size_t capacity() const { return cap_; }
    explicit operator bool() const { return ptr_ != nullptr; }
};


// ===========================================================================
// ThreadLocalBuffer — 线程本地缓冲区
// ===========================================================================
// 用途：OpenMP 并行区域中的每个线程需要独立的临时缓冲区
// 特性：
//   - thread_local 存储，无锁竞争
//   - 首次使用时分配，线程结束时释放
//   - 可复用

struct ThreadLocalBuffer {
    float* ptr = nullptr;
    size_t sz = 0;

    ThreadLocalBuffer() = default;
    ~ThreadLocalBuffer() {
        aligned_free(ptr);
    }

    // 获取至少 n 个 float 的缓冲区
    float* get(size_t n) {
        if (n > sz) {
            aligned_free(ptr);
            ptr = aligned_alloc_float(n);
            sz = n;
        }
        return ptr;
    }

    // 禁止拷贝/移动
    ThreadLocalBuffer(const ThreadLocalBuffer&) = delete;
    ThreadLocalBuffer& operator=(const ThreadLocalBuffer&) = delete;
};


// ===========================================================================
// 便捷宏：声明 thread_local 缓冲区
// ===========================================================================
// 用法：
//   #pragma omp parallel
//   {
//       THREAD_LOCAL_BUFFER(bufA, MC * KC);
//       THREAD_LOCAL_BUFFER(bufB, KC * NC);
//       // 使用 bufA, bufB ...
//   }

#define THREAD_LOCAL_BUFFER(name, size_expr) \
    thread_local ThreadLocalBuffer _tls_##name; \
    float* name = _tls_##name.get(size_expr)
