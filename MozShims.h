// Standalone shims for Mozilla types used by StreamingScaler and SkConvolver.
// This allows the benchmark to compile without the full Firefox tree.

#ifndef BENCHMARK_MOZ_SHIMS_H_
#define BENCHMARK_MOZ_SHIMS_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define MOZ_ASSERT(...) assert((__VA_ARGS__))
#define MOZ_RELEASE_ASSERT(...) assert((__VA_ARGS__))
#define MOZ_ALWAYS_INLINE inline __attribute__((always_inline))
#define MOZ_UNLIKELY(x) __builtin_expect(!!(x), 0)

// --- mozilla::gfx types ---

namespace mozilla::gfx {

enum class SurfaceFormat : int8_t {
  B8G8R8A8,
  B8G8R8X8,
  R8G8B8A8,
  R8G8B8X8,
  A8,
  UNKNOWN = -1,
};

inline bool IsOpaque(SurfaceFormat aFormat) {
  switch (aFormat) {
    case SurfaceFormat::B8G8R8X8:
    case SurfaceFormat::R8G8B8X8:
      return true;
    default:
      return false;
  }
}

static inline int BytesPerPixel(SurfaceFormat aFormat) {
  switch (aFormat) {
    case SurfaceFormat::A8:
      return 1;
    default:
      return 4;
  }
}

struct IntSize {
  int32_t width;
  int32_t height;
};

// Minimal AlignedArray matching the Firefox version.
template <typename T, int alignment = 16>
struct AlignedArray final {
  typedef T value_type;

  AlignedArray() : mPtr(nullptr), mStorage(nullptr), mCount(0) {}

  ~AlignedArray() { Dealloc(); }

  void Dealloc() {
    free(mStorage);
    mStorage = nullptr;
    mPtr = nullptr;
    mCount = 0;
  }

  void Realloc(size_t aCount, bool aZero = false) {
    free(mStorage);
    size_t bytes = sizeof(T) * aCount + (alignment - 1);
    if (aZero) {
      mStorage = static_cast<uint8_t*>(calloc(1u, bytes));
    } else {
      mStorage = static_cast<uint8_t*>(malloc(bytes));
    }
    if (!mStorage) {
      mPtr = nullptr;
      mCount = 0;
      return;
    }
    if (uintptr_t(mStorage) % alignment) {
      mPtr = (T*)(uintptr_t(mStorage) + alignment -
                  (uintptr_t(mStorage) % alignment));
    } else {
      mPtr = (T*)(mStorage);
    }
    mPtr = new (mPtr) T[aCount];
    mCount = aCount;
  }

  operator T*() { return mPtr; }
  explicit operator bool() const { return mPtr != nullptr; }

  T* mPtr;

 private:
  uint8_t* mStorage;
  size_t mCount;
};

}  // namespace mozilla::gfx

// --- mozilla::Vector shim (wraps std::vector) ---

namespace mozilla {

template <typename T, size_t InlineCapacity = 0>
class Vector {
 public:
  Vector() = default;

  size_t length() const { return mData.size(); }

  bool reserve(size_t aCapacity) {
    mData.reserve(aCapacity);
    return true;
  }

  bool resize(size_t aSize) {
    mData.resize(aSize);
    return true;
  }

  bool append(const T& aVal) {
    mData.push_back(aVal);
    return true;
  }

  bool append(const T* aBegin, size_t aLen) {
    mData.insert(mData.end(), aBegin, aBegin + aLen);
    return true;
  }

  void shrinkBy(size_t aAmount) {
    mData.resize(mData.size() - aAmount);
  }

  void shrinkTo(size_t aSize) { mData.resize(aSize); }

  T& operator[](size_t aIndex) { return mData[aIndex]; }
  const T& operator[](size_t aIndex) const { return mData[aIndex]; }

  T* begin() { return mData.data(); }
  const T* begin() const { return mData.data(); }

 private:
  std::vector<T> mData;
};

// --- CPUID helpers ---

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>

inline bool supports_sse2() { return true; }

inline bool supports_avx2() {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
  return (ebx & (1 << 5)) != 0;
}

#elif defined(__aarch64__)

inline bool supports_neon() { return true; }

#endif

}  // namespace mozilla

#endif  // BENCHMARK_MOZ_SHIMS_H_
