#include "MozShims.h"
#include "StreamingScaler.h"
#include "SkConvolver.h"

#include "png.h"
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace mozilla::gfx;

// --- Image loading ---

struct BenchImage {
  uint8_t* mBuffer;
  int mWidth;
  int mHeight;
};

static BenchImage LoadPng(const char* aPath) {
  BenchImage image;
  png_structp rpng =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!rpng) {
    fprintf(stderr, "Unable to create PNG read struct.\n");
    exit(1);
  }
  if (setjmp(png_jmpbuf(rpng))) {
    fprintf(stderr, "PNG Decoding Error.\n");
    exit(1);
  }

  FILE* input = fopen(aPath, "rb");
  if (!input) {
    fprintf(stderr, "Unable to open file: %s\n", aPath);
    exit(1);
  }

  png_infop rinfo = png_create_info_struct(rpng);
  png_init_io(rpng, input);
  png_read_info(rpng, rinfo);

  int colorType = png_get_color_type(rpng, rinfo);
  if (colorType != PNG_COLOR_TYPE_RGBA && colorType != PNG_COLOR_TYPE_RGB) {
    fprintf(stderr, "Input image must be RGB or RGBA.\n");
    exit(1);
  }

  if (colorType == PNG_COLOR_TYPE_RGB) {
    png_set_filler(rpng, 0xff, PNG_FILLER_AFTER);
  }

  image.mWidth = png_get_image_width(rpng, rinfo);
  image.mHeight = png_get_image_height(rpng, rinfo);

  size_t rowStride = image.mWidth * 4;
  size_t bufLen = static_cast<size_t>(image.mHeight) * rowStride;
  image.mBuffer = static_cast<uint8_t*>(malloc(bufLen));
  auto** bufPtrs =
      static_cast<uint8_t**>(malloc(image.mHeight * sizeof(uint8_t*)));
  if (!image.mBuffer || !bufPtrs) {
    fprintf(stderr, "Unable to allocate buffers.\n");
    exit(1);
  }

  for (int i = 0; i < image.mHeight; i++) {
    bufPtrs[i] = image.mBuffer + i * rowStride;
  }

  png_read_image(rpng, bufPtrs);
  png_destroy_read_struct(&rpng, &rinfo, nullptr);
  free(bufPtrs);
  fclose(input);
  return image;
}

static double TimeToMs(clock_t aT) {
  return static_cast<double>(aT) * 1000.0 / CLOCKS_PER_SEC;
}

// --- StreamingScaler benchmark ---

enum class SimdLevel { Scalar, Sse2, Avx2, Neon };

using SSInFn = void (*)(StreamingScaler::State*, const uint8_t*);
using SSOutFn = void (*)(StreamingScaler::State*, uint8_t*);

static clock_t ResizeStreaming(BenchImage aImage, int aOutWidth, int aOutHeight,
                               SurfaceFormat aFmt, SSInFn aIn, SSOutFn aOut) {
  StreamingScaler scaler;
  IntSize inSize{aImage.mWidth, aImage.mHeight};
  IntSize outSize{aOutWidth, aOutHeight};
  if (!scaler.Init(inSize, outSize, aFmt)) {
    fprintf(stderr, "StreamingScaler::Init failed\n");
    exit(1);
  }

  uint8_t* inbuf = aImage.mBuffer;
  size_t inRowStride = aImage.mWidth * 4;
  auto* outbuf = static_cast<uint8_t*>(malloc(aOutWidth * 4));
  if (!outbuf) {
    fprintf(stderr, "Unable to allocate output buffer.\n");
    exit(1);
  }

  StreamingScaler::State* state = scaler.GetState();
  clock_t t = clock();

  for (int i = 0; i < aOutHeight; i++) {
    while (scaler.Slots()) {
      aIn(state, inbuf);
      inbuf += inRowStride;
    }
    aOut(state, outbuf);
  }

  t = clock() - t;
  free(outbuf);
  return t;
}

// --- Lanczos3 benchmark ---

static clock_t ResizeLanczos3(BenchImage aImage, int aOutWidth, int aOutHeight,
                              SurfaceFormat aFmt,
                              skia::ConvolveSimdLevel aLevel) {
  skia::SkLanczosFilter lanczos;

  skia::SkConvolutionFilter1D xFilter;
  if (!xFilter.ComputeFilterValues(lanczos, aImage.mWidth, aOutWidth)) {
    fprintf(stderr, "Lanczos3 xFilter failed\n");
    exit(1);
  }

  skia::SkConvolutionFilter1D yFilter;
  if (!yFilter.ComputeFilterValues(lanczos, aImage.mHeight, aOutHeight)) {
    fprintf(stderr, "Lanczos3 yFilter failed\n");
    exit(1);
  }

  int srcStride = aImage.mWidth * 4;
  int dstStride = aOutWidth * 4;
  auto* output = static_cast<uint8_t*>(calloc(1, dstStride * aOutHeight));
  if (!output) {
    fprintf(stderr, "Unable to allocate output buffer.\n");
    exit(1);
  }

  clock_t t = clock();
  bool ok = skia::BGRAConvolve2DLevel(aImage.mBuffer, srcStride, aFmt, xFilter,
                                      yFilter, dstStride, output, aLevel);
  t = clock() - t;

  if (!ok) {
    fprintf(stderr, "BGRAConvolve2DLevel failed\n");
    exit(1);
  }

  free(output);
  return t;
}

// --- Benchmark driver ---

struct Impl {
  const char* mName;
  bool mIsLanczos3;
  SSInFn mIn;
  SSOutFn mOut;
  skia::ConvolveSimdLevel mL3Level;
};

static void DoBench(BenchImage aImage, double aRatio, int aIterations,
                    SurfaceFormat aFmt, Impl aImpl) {
  int outWidth = lround(aImage.mWidth * aRatio);
  int outHeight = lround(aImage.mHeight * aRatio);
  if (outWidth < 1) outWidth = 1;
  if (outHeight < 1) outHeight = 1;

  clock_t tMin = 0;
  for (int i = 0; i < aIterations; i++) {
    clock_t tTmp;
    if (aImpl.mIsLanczos3) {
      tTmp = ResizeLanczos3(aImage, outWidth, outHeight, aFmt, aImpl.mL3Level);
    } else {
      tTmp = ResizeStreaming(aImage, outWidth, outHeight, aFmt, aImpl.mIn,
                             aImpl.mOut);
    }
    if (!tMin || tTmp < tMin) {
      tMin = tTmp;
    }
  }

  printf("    to %4dx%4d %6.2fms\n", outWidth, outHeight, TimeToMs(tMin));
}

static void DoBenchSizes(BenchImage aImage, const char* aFmtName,
                         SurfaceFormat aFmt, int aIterations, Impl aImpl) {
  double ratios[] = {0.01, 0.125, 0.8};
  size_t numRatios = sizeof(ratios) / sizeof(ratios[0]);

  const char* scaler = aImpl.mIsLanczos3 ? "lanczos3" : "StreamingScaler";
  printf("%dx%d %s %s [%s]\n", aImage.mWidth, aImage.mHeight, scaler,
         aFmtName, aImpl.mName);

  for (size_t i = 0; i < numRatios; i++) {
    DoBench(aImage, ratios[i], aIterations, aFmt, aImpl);
  }
}

static void RunBench(BenchImage aImage, const char* aFmtFilter,
                     int aIterations, Impl* aImpls, int aNumImpls) {
  struct FmtEntry {
    const char* name;
    SurfaceFormat fmt;
  };

  FmtEntry formats[] = {
      {"BGRA", SurfaceFormat::B8G8R8A8},
      {"BGRX", SurfaceFormat::B8G8R8X8},
  };
  size_t numFormats = sizeof(formats) / sizeof(formats[0]);

  if (aFmtFilter) {
    size_t i;
    for (i = 0; i < numFormats; i++) {
      if (strcmp(formats[i].name, aFmtFilter) == 0) break;
    }
    if (i >= numFormats) {
      fprintf(stderr, "Format not recognized. Use BGRA or BGRX.\n");
      exit(1);
    }
    for (int j = 0; j < aNumImpls; j++) {
      DoBenchSizes(aImage, formats[i].name, formats[i].fmt, aIterations,
                   aImpls[j]);
    }
    return;
  }

  for (size_t i = 0; i < numFormats; i++) {
    for (int j = 0; j < aNumImpls; j++) {
      DoBenchSizes(aImage, formats[i].name, formats[i].fmt, aIterations,
                   aImpls[j]);
    }
  }
}

int main(int argc, char* argv[]) {
  int argPos = 1;
  int implMode = 0;  // 0=all, 1=scalar, 3=sse2, 4=avx2, 5=neon
  const char* scalerFilter = nullptr;

  while (argPos < argc && argv[argPos][0] == '-') {
    if (strcmp(argv[argPos], "--scalar") == 0) {
      implMode = 1;
    } else if (strcmp(argv[argPos], "--sse2") == 0) {
      implMode = 3;
    } else if (strcmp(argv[argPos], "--avx2") == 0) {
      implMode = 4;
    } else if (strcmp(argv[argPos], "--neon") == 0) {
      implMode = 5;
    } else if (strcmp(argv[argPos], "--streaming") == 0) {
      scalerFilter = "streaming";
    } else if (strcmp(argv[argPos], "--lanczos3") == 0) {
      scalerFilter = "lanczos3";
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[argPos]);
      return 1;
    }
    argPos++;
  }

  int remaining = argc - argPos;
  if (remaining < 1 || remaining > 2) {
    fprintf(stderr,
            "Usage: %s [--scalar|--sse2|--avx2|--neon] "
            "[--streaming|--lanczos3] <path.png> [BGRA|BGRX]\n",
            argv[0]);
    return 1;
  }

  const char* path = argv[argPos];
  const char* fmtArg = (remaining == 2) ? argv[argPos + 1] : nullptr;

  int iterations = 100;
  if (getenv("ITERATIONS")) {
    char* end;
    errno = 0;
    unsigned long ul = strtoul(getenv("ITERATIONS"), &end, 10);
    if (*end != '\0' || errno != 0 || ul == 0 || ul > INT_MAX) {
      fprintf(stderr, "Invalid ITERATIONS env var.\n");
      return 1;
    }
    iterations = static_cast<int>(ul);
  }
  fprintf(stderr, "Iterations: %d\n", iterations);

  BenchImage image = LoadPng(path);

  Impl impls[10];
  int numImpls = 0;

  // --- StreamingScaler implementations ---
  if (!scalerFilter || strcmp(scalerFilter, "streaming") == 0) {
    if (implMode == 0 || implMode == 1) {
      impls[numImpls].mName = "scalar";
      impls[numImpls].mIsLanczos3 = false;
      impls[numImpls].mIn = StreamingScaler::InScalar;
      impls[numImpls].mOut = StreamingScaler::OutScalar;
      numImpls++;
    }

#if defined(__x86_64__)
    if (implMode == 0 || implMode == 3) {
      impls[numImpls].mName = "sse2";
      impls[numImpls].mIsLanczos3 = false;
      impls[numImpls].mIn = StreamingScaler::InSse2;
      impls[numImpls].mOut = StreamingScaler::OutSse2;
      numImpls++;
    }
    if (implMode == 0 || implMode == 4) {
      impls[numImpls].mName = "avx2";
      impls[numImpls].mIsLanczos3 = false;
      impls[numImpls].mIn = StreamingScaler::InAvx2;
      impls[numImpls].mOut = StreamingScaler::OutAvx2;
      numImpls++;
    }
    if (implMode == 5) {
      fprintf(stderr, "NEON not available on x86_64.\n");
      return 1;
    }
#elif defined(__aarch64__)
    if (implMode == 0 || implMode == 5) {
      impls[numImpls].mName = "neon";
      impls[numImpls].mIsLanczos3 = false;
      impls[numImpls].mIn = StreamingScaler::InNeon;
      impls[numImpls].mOut = StreamingScaler::OutNeon;
      numImpls++;
    }
    if (implMode == 3 || implMode == 4) {
      fprintf(stderr, "SSE2/AVX2 not available on AArch64.\n");
      return 1;
    }
#else
    if (implMode >= 3) {
      fprintf(stderr, "No SIMD support compiled in.\n");
      return 1;
    }
#endif
  }

  // --- Lanczos3 implementations (per SIMD level) ---
  if (!scalerFilter || strcmp(scalerFilter, "lanczos3") == 0) {
    if (implMode == 0 || implMode == 1) {
      impls[numImpls].mName = "scalar";
      impls[numImpls].mIsLanczos3 = true;
      impls[numImpls].mIn = nullptr;
      impls[numImpls].mOut = nullptr;
      impls[numImpls].mL3Level = skia::ConvolveSimdLevel::Scalar;
      numImpls++;
    }

#if defined(__x86_64__)
    if (implMode == 0 || implMode == 3) {
      impls[numImpls].mName = "sse2";
      impls[numImpls].mIsLanczos3 = true;
      impls[numImpls].mIn = nullptr;
      impls[numImpls].mOut = nullptr;
      impls[numImpls].mL3Level = skia::ConvolveSimdLevel::Sse2;
      numImpls++;
    }
    if (implMode == 0 || implMode == 4) {
      // Note: no AVX2 horizontal exists, so this is SSE2 horiz + AVX2 vert.
      impls[numImpls].mName = "avx2";
      impls[numImpls].mIsLanczos3 = true;
      impls[numImpls].mIn = nullptr;
      impls[numImpls].mOut = nullptr;
      impls[numImpls].mL3Level = skia::ConvolveSimdLevel::Avx2;
      numImpls++;
    }
#elif defined(__aarch64__)
    if (implMode == 0 || implMode == 5) {
      impls[numImpls].mName = "neon";
      impls[numImpls].mIsLanczos3 = true;
      impls[numImpls].mIn = nullptr;
      impls[numImpls].mOut = nullptr;
      impls[numImpls].mL3Level = skia::ConvolveSimdLevel::Neon;
      numImpls++;
    }
#endif
  }

  if (numImpls == 0) {
    fprintf(stderr, "No implementations to benchmark.\n");
    return 1;
  }

  RunBench(image, fmtArg, iterations, impls, numImpls);
  free(image.mBuffer);
  return 0;
}
