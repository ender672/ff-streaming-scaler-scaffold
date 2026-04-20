#include "MozShims.h"
#include "StreamingScaler.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace mozilla::gfx;

using SSInFn = void (*)(StreamingScaler::State*, const uint8_t*);
using SSOutFn = void (*)(StreamingScaler::State*, uint8_t*);

static SSInFn gCurIn;
static SSOutFn gCurOut;

static constexpr int kCmp = 4;

// --- Reference helpers (long double precision) ---

static long double Cubic(long double b, long double c, long double x) {
  if (x < 1.0L) {
    return ((12.0L - 9.0L * b - 6.0L * c) * x * x * x +
            (-18.0L + 12.0L * b + 6.0L * c) * x * x + (6.0L - 2.0L * b)) /
           6.0L;
  }
  if (x < 2.0L) {
    return ((-b - 6.0L * c) * x * x * x +
            (6.0L * b + 30.0L * c) * x * x +
            (-12.0L * b - 48.0L * c) * x + (8.0L * b + 24.0L * c)) /
           6.0L;
  }
  return 0.0L;
}

static long double RefCatrom(long double x) { return Cubic(0, 0.5L, x); }

static int RefMaxTapsCheck(int aDimIn, int aDimOut) {
  if (aDimIn <= aDimOut) {
    return 4;
  }
  return static_cast<int>(ceill(4.0L * aDimIn / aDimOut)) + 1;
}

static long double RefMap(int aDimIn, int aDimOut, int aPos) {
  return (aPos + 0.5L) * (long double)aDimIn / aDimOut - 0.5L;
}

static int RefSplitMap(int aDimIn, int aDimOut, int aPos, long double* aTy) {
  long double smp = RefMap(aDimIn, aDimOut, aPos);
  int smpI = floorl(smp);
  *aTy = smp - smpI;
  return smpI;
}

static void RefCalcCoeffs(long double* aCoeffs, int aNSamples, int aSmpStart,
                          long double aCenter, long double aTapMult) {
  assert(aNSamples > 0);
  long double fudge = 0.0L;
  for (int i = 0; i < aNSamples; i++) {
    long double dist = fabsl((long double)(aSmpStart + i) - aCenter);
    aCoeffs[i] = RefCatrom(dist / aTapMult) / aTapMult;
    fudge += aCoeffs[i];
  }
  for (int i = 0; i < aNSamples; i++) {
    aCoeffs[i] /= fudge;
  }
}

static void FillRand8(uint8_t* aBuf, int aLen) {
  for (int i = 0; i < aLen; i++) {
    aBuf[i] = rand() % 256;
  }
}

// --- 2D allocation helpers ---

static uint8_t** Alloc2dU8(int aWidth, int aHeight) {
  auto** rows = static_cast<uint8_t**>(malloc(aHeight * sizeof(uint8_t*)));
  for (int i = 0; i < aHeight; i++) {
    rows[i] = static_cast<uint8_t*>(malloc(aWidth * sizeof(uint8_t)));
  }
  return rows;
}

static void Free2dU8(uint8_t** aPtr, int aHeight) {
  for (int i = 0; i < aHeight; i++) {
    free(aPtr[i]);
  }
  free(aPtr);
}

static long double** Alloc2dLd(int aWidth, int aHeight) {
  auto** rows =
      static_cast<long double**>(malloc(aHeight * sizeof(long double*)));
  for (int i = 0; i < aHeight; i++) {
    rows[i] =
        static_cast<long double*>(malloc(aWidth * sizeof(long double)));
  }
  return rows;
}

static void Free2dLd(long double** aPtr, int aHeight) {
  for (int i = 0; i < aHeight; i++) {
    free(aPtr[i]);
  }
  free(aPtr);
}

// --- Reference scaling ---

static void RefXScale(long double* aIn, int aInWidth, long double* aOut,
                      int aOutWidth) {
  long double tapMult =
      aInWidth <= aOutWidth ? 1.0L : (long double)aInWidth / aOutWidth;
  long double radius = 2.0L * tapMult;
  auto* coeffs = static_cast<long double*>(
      malloc(RefMaxTapsCheck(aInWidth, aOutWidth) * sizeof(long double)));
  int maxPos = aInWidth - 1;

  for (int i = 0; i < aOutWidth; i++) {
    long double tx;
    int smpI = RefSplitMap(aInWidth, aOutWidth, i, &tx);
    long double center = smpI + tx;

    int smpStart = static_cast<int>(ceill(center - radius));
    int smpEnd = static_cast<int>(floorl(center + radius));
    if ((long double)smpStart == center - radius) {
      smpStart++;
    }
    if ((long double)smpEnd == center + radius) {
      smpEnd--;
    }
    if (smpStart < 0) {
      smpStart = 0;
    }
    if (smpEnd > maxPos) {
      smpEnd = maxPos;
    }
    int nSamples = smpEnd - smpStart + 1;

    RefCalcCoeffs(coeffs, nSamples, smpStart, center, tapMult);

    for (int j = 0; j < kCmp; j++) {
      aOut[i * kCmp + j] = 0;
      for (int k = 0; k < nSamples; k++) {
        aOut[i * kCmp + j] += coeffs[k] * aIn[(smpStart + k) * kCmp + j];
      }
    }
  }

  free(coeffs);
}

static void RefTransposeLine(long double* aIn, int aWidth, long double** aOut,
                             int aOutOffset) {
  for (int i = 0; i < aWidth; i++) {
    for (int j = 0; j < kCmp; j++) {
      aOut[i][aOutOffset + j] = aIn[i * kCmp + j];
    }
  }
}

static void RefTransposeColumn(long double** aIn, int aHeight, long double* aOut,
                               int aInOffset) {
  for (int i = 0; i < aHeight; i++) {
    for (int j = 0; j < kCmp; j++) {
      aOut[i * kCmp + j] = aIn[i][aInOffset + j];
    }
  }
}

static void RefYScale(long double** aIn, int aWidth, int aInHeight,
                      long double** aOut, int aOutHeight) {
  auto* transposed =
      static_cast<long double*>(malloc(aInHeight * kCmp * sizeof(long double)));
  auto* transScaled =
      static_cast<long double*>(malloc(aOutHeight * kCmp * sizeof(long double)));

  for (int i = 0; i < aWidth; i++) {
    RefTransposeColumn(aIn, aInHeight, transposed, i * kCmp);
    RefXScale(transposed, aInHeight, transScaled, aOutHeight);
    RefTransposeLine(transScaled, aOutHeight, aOut, i * kCmp);
  }

  free(transposed);
  free(transScaled);
}

static long double ClampLd(long double aIn) {
  if (aIn <= 0.0L) return 0.0L;
  if (aIn >= 1.0L) return 1.0L;
  return aIn;
}

static void RefScale(uint8_t** aIn, int aInWidth, int aInHeight,
                     long double** aOut, int aOutWidth, int aOutHeight,
                     bool aHasAlpha) {
  int stride = kCmp * aInWidth;

  auto* preLine =
      static_cast<long double*>(malloc(stride * sizeof(long double)));
  long double** intermediate = Alloc2dLd(aOutWidth * kCmp, aInHeight);

  for (int i = 0; i < aInHeight; i++) {
    for (int j = 0; j < stride; j++) {
      preLine[j] = aIn[i][j] / 255.0L;
    }

    // For BGRX, alpha is ignored (forced to 1.0 later).
    // For BGRA, all four channels are scaled independently.

    RefXScale(preLine, aInWidth, intermediate[i], aOutWidth);
  }

  RefYScale(intermediate, aOutWidth, aInHeight, aOut, aOutHeight);

  for (int i = 0; i < aOutHeight; i++) {
    for (int j = 0; j < aOutWidth; j++) {
      long double* px = aOut[i] + j * kCmp;
      px[0] = ClampLd(px[0]);
      px[1] = ClampLd(px[1]);
      px[2] = ClampLd(px[2]);
      if (aHasAlpha) {
        long double alpha = ClampLd(px[3]);
        // Enforce alpha >= max(R,G,B)
        long double maxRgb = px[0];
        if (px[1] > maxRgb) maxRgb = px[1];
        if (px[2] > maxRgb) maxRgb = px[2];
        px[3] = (alpha > maxRgb) ? alpha : maxRgb;
      } else {
        px[3] = 1.0L;
      }
    }
  }

  free(preLine);
  Free2dLd(intermediate, aInHeight);
}

// --- Validation ---

static double gWorst;

static void ValidateScanline(uint8_t* aOil, long double* aRef, int aWidth) {
  for (int i = 0; i < aWidth; i++) {
    for (int j = 0; j < kCmp; j++) {
      int pos = i * kCmp + j;
      double refF = aRef[pos] * 255.0;
      double error = fabs(aOil[pos] - refF) - 0.5;
      if (error > gWorst) {
        gWorst = error;
      }
      if (error > 0.07) {
        int refI = lround(refF);
        fprintf(stderr, "[%d:%d] expected: %d, got %d (%.9f)\n", i, j, refI,
                aOil[pos], refF);
        assert(0 && "pixel error exceeds tolerance");
      }
    }
  }
}

// --- StreamingScaler driver ---

static void DoScale(uint8_t** aInput, int aInWidth, int aInHeight,
                    uint8_t** aOutput, int aOutWidth, int aOutHeight,
                    SurfaceFormat aFmt) {
  StreamingScaler scaler;
  IntSize inSize{aInWidth, aInHeight};
  IntSize outSize{aOutWidth, aOutHeight};
  if (!scaler.Init(inSize, outSize, aFmt)) {
    fprintf(stderr, "StreamingScaler::Init failed\n");
    exit(1);
  }

  StreamingScaler::State* state = scaler.GetState();
  int inLine = 0;
  for (int i = 0; i < aOutHeight; i++) {
    while (scaler.Slots()) {
      gCurIn(state, aInput[inLine++]);
    }
    gCurOut(state, aOutput[i]);
  }
}

static void TestScale(int aInWidth, int aInHeight, uint8_t** aInput,
                      int aOutWidth, int aOutHeight, SurfaceFormat aFmt) {
  bool hasAlpha = !IsOpaque(aFmt);
  int outRowStride = kCmp * aOutWidth;

  uint8_t** oilOutput = Alloc2dU8(outRowStride, aOutHeight);
  DoScale(aInput, aInWidth, aInHeight, oilOutput, aOutWidth, aOutHeight, aFmt);

  long double** refOutput = Alloc2dLd(outRowStride, aOutHeight);
  RefScale(aInput, aInWidth, aInHeight, refOutput, aOutWidth, aOutHeight,
           hasAlpha);

  for (int i = 0; i < aOutHeight; i++) {
    ValidateScanline(oilOutput[i], refOutput[i], aOutWidth);
  }

  Free2dU8(oilOutput, aOutHeight);
  Free2dLd(refOutput, aOutHeight);
}

static void TestScaleSquareRand(int aInDim, int aOutDim, SurfaceFormat aFmt) {
  int inRowStride = kCmp * aInDim;

  uint8_t** input = Alloc2dU8(inRowStride, aInDim);
  for (int i = 0; i < aInDim; i++) {
    FillRand8(input[i], inRowStride);
  }
  TestScale(aInDim, aInDim, input, aOutDim, aOutDim, aFmt);
  Free2dU8(input, aInDim);
}

static void TestScaleCatromExtremes() {
  uint8_t** input = Alloc2dU8(4 * kCmp, 4);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4 * kCmp; j++) {
      input[i][j] = 0;
    }
  }

  // Set a bright 2x2 block in the center
  for (int i = 1; i <= 2; i++) {
    for (int j = 1; j <= 2; j++) {
      for (int k = 0; k < kCmp; k++) {
        input[i][j * kCmp + k] = 255;
      }
    }
  }

  TestScale(4, 4, input, 3, 3, SurfaceFormat::B8G8R8X8);
  Free2dU8(input, 4);
}

static void TestScaleEachFmt(int aDimA, int aDimB) {
  TestScaleSquareRand(aDimA, aDimB, SurfaceFormat::B8G8R8A8);
  TestScaleSquareRand(aDimA, aDimB, SurfaceFormat::B8G8R8X8);
}

static void TestScaleAll() {
  TestScaleEachFmt(5, 1);
  TestScaleEachFmt(8, 1);
  TestScaleEachFmt(8, 3);
  TestScaleEachFmt(100, 1);
  TestScaleEachFmt(100, 99);
  TestScaleEachFmt(2, 1);
}

// --- Rectangular (non-square) tests ---

static void TestScaleRect(int aInW, int aInH, int aOutW, int aOutH,
                          SurfaceFormat aFmt) {
  int inRowStride = kCmp * aInW;

  uint8_t** input = Alloc2dU8(inRowStride, aInH);
  for (int i = 0; i < aInH; i++) {
    FillRand8(input[i], inRowStride);
  }
  TestScale(aInW, aInH, input, aOutW, aOutH, aFmt);
  Free2dU8(input, aInH);
}

static void TestScaleRectAll() {
  TestScaleRect(200, 100, 50, 25, SurfaceFormat::B8G8R8A8);
  TestScaleRect(200, 100, 50, 25, SurfaceFormat::B8G8R8X8);
  TestScaleRect(100, 200, 25, 50, SurfaceFormat::B8G8R8A8);
  TestScaleRect(100, 200, 25, 50, SurfaceFormat::B8G8R8X8);
  TestScaleRect(640, 480, 160, 120, SurfaceFormat::B8G8R8A8);
  TestScaleRect(640, 480, 160, 120, SurfaceFormat::B8G8R8X8);
}

// --- Near-identity precision sweep ---

/* Sweep near-identity up/down scales (N <-> N-1) across sizes, colorspaces,
 * and seeds. Targets the regime where float accumulation in coefficient
 * sums has the least headroom. */
static void TestScaleNearIdentity() {
  static const int sizes[] = {7, 16, 33, 50, 99, 100};
  static const SurfaceFormat spaces[] = {SurfaceFormat::B8G8R8A8,
                                         SurfaceFormat::B8G8R8X8};
  static const unsigned int seeds[] = {1531289551u, 0xdeadbeefu};
  int nSizes = sizeof(sizes) / sizeof(sizes[0]);
  int nSpaces = sizeof(spaces) / sizeof(spaces[0]);
  int nSeeds = sizeof(seeds) / sizeof(seeds[0]);

  for (int seedI = 0; seedI < nSeeds; seedI++) {
    srand(seeds[seedI]);
    for (int szI = 0; szI < nSizes; szI++) {
      int n = sizes[szI];
      for (int csI = 0; csI < nSpaces; csI++) {
        SurfaceFormat cs = spaces[csI];
        // StreamingScaler only supports downscale; only test N -> N-1.
        if (n > 1) {
          TestScaleSquareRand(n, n - 1, cs);
        }
      }
    }
  }
}

// --- Reset tests ---

static void TestReset(int aInW, int aInH, int aOutW, int aOutH,
                      SurfaceFormat aFmt) {
  int inRowStride = kCmp * aInW;
  int outRowStride = kCmp * aOutW;

  uint8_t** input = Alloc2dU8(inRowStride, aInH);
  for (int i = 0; i < aInH; i++) {
    FillRand8(input[i], inRowStride);
  }

  StreamingScaler scaler;
  IntSize inSize{aInW, aInH};
  IntSize outSize{aOutW, aOutH};
  assert(scaler.Init(inSize, outSize, aFmt));

  // First pass
  uint8_t** out1 = Alloc2dU8(outRowStride, aOutH);
  StreamingScaler::State* state = scaler.GetState();
  int inLine = 0;
  for (int i = 0; i < aOutH; i++) {
    while (scaler.Slots()) {
      gCurIn(state, input[inLine++]);
    }
    gCurOut(state, out1[i]);
  }

  // Reset and scale again
  scaler.Reset();
  state = scaler.GetState();
  uint8_t** out2 = Alloc2dU8(outRowStride, aOutH);
  inLine = 0;
  for (int i = 0; i < aOutH; i++) {
    while (scaler.Slots()) {
      gCurIn(state, input[inLine++]);
    }
    gCurOut(state, out2[i]);
  }

  // Compare: must be identical
  for (int i = 0; i < aOutH; i++) {
    for (int j = 0; j < outRowStride; j++) {
      if (out1[i][j] != out2[i][j]) {
        fprintf(stderr, "Reset mismatch at row %d col %d: %d vs %d\n", i, j,
                out1[i][j], out2[i][j]);
        assert(0 && "Reset produced different output");
      }
    }
  }

  Free2dU8(out1, aOutH);
  Free2dU8(out2, aOutH);
  Free2dU8(input, aInH);
}

static void TestResetAll() {
  TestReset(100, 100, 50, 50, SurfaceFormat::B8G8R8A8);
  TestReset(100, 100, 50, 50, SurfaceFormat::B8G8R8X8);
  TestReset(200, 100, 50, 25, SurfaceFormat::B8G8R8A8);
  TestReset(8, 8, 3, 3, SurfaceFormat::B8G8R8A8);
  TestReset(640, 480, 160, 120, SurfaceFormat::B8G8R8X8);
}

// --- Test runner ---

struct Impl {
  const char* mName;
  SSInFn mIn;
  SSOutFn mOut;
};

static void RunTests(Impl* aImpl) {
  printf("--- testing %s ---\n", aImpl->mName);
  gCurIn = aImpl->mIn;
  gCurOut = aImpl->mOut;

  TestScaleAll();
  TestScaleCatromExtremes();
  TestScaleRectAll();
  TestResetAll();
  TestScaleNearIdentity();
}

int main() {
  int seed = 1531289551;
  printf("seed: %d\n", seed);
  srand(seed);

  Impl impls[4];
  int numImpls = 0;

  impls[numImpls].mName = "scalar";
  impls[numImpls].mIn = StreamingScaler::InScalar;
  impls[numImpls].mOut = StreamingScaler::OutScalar;
  numImpls++;

#if defined(__x86_64__)
  impls[numImpls].mName = "sse2";
  impls[numImpls].mIn = StreamingScaler::InSse2;
  impls[numImpls].mOut = StreamingScaler::OutSse2;
  numImpls++;

  impls[numImpls].mName = "avx2";
  impls[numImpls].mIn = StreamingScaler::InAvx2;
  impls[numImpls].mOut = StreamingScaler::OutAvx2;
  numImpls++;
#elif defined(__aarch64__)
  impls[numImpls].mName = "neon";
  impls[numImpls].mIn = StreamingScaler::InNeon;
  impls[numImpls].mOut = StreamingScaler::OutNeon;
  numImpls++;
#endif

  for (int i = 0; i < numImpls; i++) {
    RunTests(&impls[i]);
  }

  printf("worst error: %f\n", gWorst);
  printf("All tests pass.\n");
  return 0;
}
