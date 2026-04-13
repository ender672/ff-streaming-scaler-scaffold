/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "StreamingScaler.h"

#include <algorithm>
#include <cmath>

#include "StreamingScalerInternal.h"

#ifdef USE_SSE2
#include "MozShims.h"
#endif

#ifdef USE_NEON
#include "MozShims.h"
#endif

namespace mozilla::gfx {

/**
 * When shrinking a 10 million pixel wide scanline down to a single pixel, we
 * reach the limits of single-precision floats. Limit input dimensions to one
 * million by one million pixels to avoid this issue as well as overflow issues
 * with 32-bit ints.
 */
static constexpr int kMaxDimension = 1000000;

/**
 * Bicubic interpolation. 2 base taps on either side.
 */
static constexpr int kTaps = 4;

static float ClampF(float aX) { return std::clamp(aX, 0.0f, 1.0f); }

/**
 * Map from the discreet dest coordinate pos to a continuous source coordinate.
 * The resulting coordinate can range from -0.5 to the maximum of the
 * destination image dimension.
 */
static double Map(int aDimIn, int aDimOut, int aPos) {
  return (aPos + 0.5) * (static_cast<double>(aDimIn) / aDimOut) - 0.5;
}

/**
 * Returns the mapped input position and put the sub-pixel remainder in rest.
 */
static int SplitMap(int aDimIn, int aDimOut, int aPos, float* aRest) {
  double smp = Map(aDimIn, aDimOut, aPos);
  int smpI = smp < 0 ? -1 : smp;
  *aRest = smp - smpI;
  return smpI;
}

/**
 * Given input and output dimension, calculate the total number of taps that
 * will be needed to calculate an output sample.
 *
 * When we reduce an image by a factor of two, we need to scale our resampling
 * function by two as well in order to avoid aliasing.
 */
static int CalcTaps(int aDimIn, int aDimOut) {
  if (aDimOut > aDimIn) {
    return kTaps;
  }
  int tmp = kTaps * aDimIn / aDimOut;
  return tmp - (tmp & 1);
}

/**
 * Catmull-Rom interpolator.
 */
static float Catrom(float aX) {
  if (aX < 1) {
    return (1.5f * aX - 2.5f) * aX * aX + 1;
  }
  return (((5 - aX) * aX - 8) * aX + 4) / 2;
}

/**
 * Given an offset tx, calculate taps coefficients.
 */
static void CalcCoeffs(float* aCoeffs, float aTx, int aTaps, int aLtrim,
                       int aRtrim) {
  float tapMult = static_cast<float>(aTaps) / kTaps;
  aTx = 1 - aTx - aTaps / 2 + aLtrim;
  float fudge = 0.0f;

  for (int i = aLtrim; i < aTaps - aRtrim; i++) {
    float tmp = Catrom(fabsf(aTx) / tapMult) / tapMult;
    fudge += tmp;
    aCoeffs[i] = tmp;
    aTx += 1;
  }
  fudge = 1 / fudge;
  for (int i = aLtrim; i < aTaps - aRtrim; i++) {
    aCoeffs[i] *= fudge;
  }
}

/**
 * Takes a sample value, an array of 4 coefficients & 4 accumulators, and
 * adds the product of sample * coeffs[n] to each accumulator.
 */
static void AddSampleToSumF(float aSample, float* aCoeffs, float* aSum) {
  for (int i = 0; i < 4; i++) {
    aSum[i] += aSample * aCoeffs[i];
  }
}

/**
 * Takes an array of 4 floats and shifts them left. The rightmost element is
 * set to 0.0.
 */
static void ShiftLeftF(float* aF) {
  aF[0] = aF[1];
  aF[1] = aF[2];
  aF[2] = aF[3];
  aF[3] = 0.0f;
}

static void YScaleOutBgra(float* aSums, int aWidth, uint8_t* aOut, int aTap) {
  int tapOff = aTap * 4;
  for (int i = 0; i < aWidth; i++) {
    uint8_t maxRgb = 0;
    for (int j = 0; j < 3; j++) {
      aOut[j] = static_cast<int>(roundf(ClampF(aSums[tapOff + j]) * 255.0f));
      if (aOut[j] > maxRgb) {
        maxRgb = aOut[j];
      }
      aSums[tapOff + j] = 0.0f;
    }
    uint8_t alpha =
        static_cast<int>(roundf(ClampF(aSums[tapOff + 3]) * 255.0f));
    aOut[3] = std::max(alpha, maxRgb);
    aSums[tapOff + 3] = 0.0f;
    aSums += 16;
    aOut += 4;
  }
}

static void YScaleOutBgrx(float* aSums, int aWidth, uint8_t* aOut, int aTap) {
  int tapOff = aTap * 4;
  for (int i = 0; i < aWidth; i++) {
    for (int j = 0; j < 3; j++) {
      aOut[j] = static_cast<int>(roundf(ClampF(aSums[tapOff + j]) * 255.0f));
      aSums[tapOff + j] = 0.0f;
    }
    aOut[3] = 255;
    aSums[tapOff + 3] = 0.0f;
    aSums += 16;
    aOut += 4;
  }
}

static void YScaleOut(float* aSums, int aWidth, uint8_t* aOut, bool aHasAlpha,
                      int aTap) {
  if (aHasAlpha) {
    YScaleOutBgra(aSums, aWidth, aOut, aTap);
  } else {
    YScaleOutBgrx(aSums, aWidth, aOut, aTap);
  }
}

/* horizontal scaling */

/**
 * Given input & output dimensions, populate a buffer of coefficients and
 * border counters.
 *
 * This method assumes that in_dim >= out_dim.
 *
 * It generates 4 * in_dim coefficients -- 4 for every input sample.
 *
 * It generates out_dim border counters, these indicate how many input samples
 * to process before the next output sample is finished.
 */
static void ScaleDownCoeffs(int aInDim, int aOutDim, float* aCoeffBuf,
                            int* aBorderBuf, float* aTmpCoeffs) {
  int taps = CalcTaps(aInDim, aOutDim);
  int ends[4] = {-1, -1, -1, -1};

  for (int i = 0; i < aOutDim; i++) {
    float tx;
    int smpI = SplitMap(aInDim, aOutDim, i, &tx);

    int smpStart = smpI - (taps / 2 - 1);
    int smpEnd = smpI + taps / 2;
    if (smpEnd >= aInDim) {
      smpEnd = aInDim - 1;
    }
    ends[i % 4] = smpEnd;
    aBorderBuf[i] = smpEnd - ends[(i + 3) % 4];

    int ltrim = 0;
    if (smpStart < 0) {
      ltrim = -1 * smpStart;
    }
    int rtrim = smpStart + (taps - 1) - smpEnd;
    CalcCoeffs(aTmpCoeffs, tx, taps, ltrim, rtrim);

    for (int j = ltrim; j < taps - rtrim; j++) {
      int pos = smpStart + j;

      int offset = 3;
      if (pos > ends[(i + 3) % 4]) {
        offset = 0;
      } else if (pos > ends[(i + 2) % 4]) {
        offset = 1;
      } else if (pos > ends[(i + 1) % 4]) {
        offset = 2;
      }

      aCoeffBuf[pos * 4 + offset] = aTmpCoeffs[j];
    }
  }
}

static void ScaleDownBgra(const uint8_t* aIn, float* aSumsY, int aOutWidth,
                          float* aCoeffsX, int* aBorderBuf, float* aCoeffsY,
                          int aTap) {
  float sum[4][4] = {{0.0f}};

  for (int i = 0; i < aOutWidth; i++) {
    for (int j = 0; j < aBorderBuf[i]; j++) {
      for (int k = 0; k < 4; k++) {
        AddSampleToSumF(gI2fMap[aIn[k]], aCoeffsX, sum[k]);
      }
      aIn += 4;
      aCoeffsX += 4;
    }

    {
      float samples[4];
      for (int j = 0; j < 4; j++) {
        samples[j] = sum[j][0];
        ShiftLeftF(sum[j]);
      }
      for (int j = 0; j < 4; j++) {
        float cy = aCoeffsY[j];
        int off = ((aTap + j) & 3) * 4;
        aSumsY[off + 0] += samples[0] * cy;
        aSumsY[off + 1] += samples[1] * cy;
        aSumsY[off + 2] += samples[2] * cy;
        aSumsY[off + 3] += samples[3] * cy;
      }
      aSumsY += 16;
    }
  }
}

static void ScaleDownBgrx(const uint8_t* aIn, float* aSumsY, int aOutWidth,
                          float* aCoeffsX, int* aBorderBuf, float* aCoeffsY,
                          int aTap) {
  float sum[4][4] = {{0.0f}};

  for (int i = 0; i < aOutWidth; i++) {
    for (int j = 0; j < aBorderBuf[i]; j++) {
      for (int k = 0; k < 3; k++) {
        AddSampleToSumF(gI2fMap[aIn[k]], aCoeffsX, sum[k]);
      }
      AddSampleToSumF(1.0f, aCoeffsX, sum[3]);
      aIn += 4;
      aCoeffsX += 4;
    }

    {
      float samples[4];
      for (int j = 0; j < 4; j++) {
        samples[j] = sum[j][0];
        ShiftLeftF(sum[j]);
      }
      for (int j = 0; j < 4; j++) {
        float cy = aCoeffsY[j];
        int off = ((aTap + j) & 3) * 4;
        aSumsY[off + 0] += samples[0] * cy;
        aSumsY[off + 1] += samples[1] * cy;
        aSumsY[off + 2] += samples[2] * cy;
        aSumsY[off + 3] += samples[3] * cy;
      }
      aSumsY += 16;
    }
  }
}

void StreamingScaler::InitCoefficients() {
  mState.mCoeffsX = mCoeffsX;
  mState.mBordersX = mBordersX;
  mState.mCoeffsY = mCoeffsY;
  mState.mBordersY = mBordersY;
  mState.mSumsY = mSumsY;
  mState.mTmpCoeffs = mTmpCoeffs;

  ScaleDownCoeffs(mState.mInWidth, mState.mOutWidth, mState.mCoeffsX,
                  mState.mBordersX, mState.mTmpCoeffs);
  ScaleDownCoeffs(mState.mInHeight, mState.mOutHeight, mState.mCoeffsY,
                  mState.mBordersY, mState.mTmpCoeffs);
}

void StreamingScaler::ScaleInputRow(const uint8_t* aIn) {
  float* coeffsY = mState.mCoeffsY + mState.mInPos * 4;

  if (mState.mHasAlpha) {
    ScaleDownBgra(aIn, mState.mSumsY, mState.mOutWidth, mState.mCoeffsX,
                  mState.mBordersX, coeffsY, mState.mSumsYTap);
  } else {
    ScaleDownBgrx(aIn, mState.mSumsY, mState.mOutWidth, mState.mCoeffsX,
                  mState.mBordersX, coeffsY, mState.mSumsYTap);
  }

  mState.mBordersY[mState.mOutPos] -= 1;
  mState.mInPos++;
}

bool StreamingScaler::Init(const IntSize& aInputSize,
                           const IntSize& aOutputSize, SurfaceFormat aFormat) {
  mCoeffsX.Dealloc();
  mBordersX.Dealloc();
  mCoeffsY.Dealloc();
  mBordersY.Dealloc();
  mSumsY.Dealloc();
  mTmpCoeffs.Dealloc();
  mState = {};

  switch (aFormat) {
    case SurfaceFormat::B8G8R8A8:
    case SurfaceFormat::B8G8R8X8:
    case SurfaceFormat::R8G8B8A8:
    case SurfaceFormat::R8G8B8X8:
      break;
    default:
      return false;
  }

  int32_t inW = aInputSize.width;
  int32_t inH = aInputSize.height;
  int32_t outW = aOutputSize.width;
  int32_t outH = aOutputSize.height;

  if (inH > kMaxDimension || outH > kMaxDimension || inH < 1 || outH < 1 ||
      inW > kMaxDimension || outW > kMaxDimension || inW < 1 || outW < 1) {
    return false;
  }

  if (outH > inH || outW > inW) {
    return false;
  }

  int tapsX = CalcTaps(inW, outW);
  int tapsY = CalcTaps(inH, outH);

  mCoeffsX.Realloc(kTaps * std::max(inW, outW), true);
  mBordersX.Realloc(std::min(inW, outW), true);
  mCoeffsY.Realloc(kTaps * std::max(inH, outH), true);
  mBordersY.Realloc(std::min(inH, outH), true);
  mSumsY.Realloc(outW * 4 * kTaps, true);
  mTmpCoeffs.Realloc(std::max(tapsX, tapsY), true);

  if (!mCoeffsX || !mBordersX || !mCoeffsY || !mBordersY || !mSumsY ||
      !mTmpCoeffs) {
    return false;
  }

  mState.mInHeight = inH;
  mState.mOutHeight = outH;
  mState.mInWidth = inW;
  mState.mOutWidth = outW;
  mState.mHasAlpha = !IsOpaque(aFormat);

  InitCoefficients();
  return true;
}

int StreamingScaler::Slots() const {
  MOZ_ASSERT(mBordersY);
  return mState.mBordersY[mState.mOutPos];
}

void StreamingScaler::FeedRow(const uint8_t* aInputRow) {
  MOZ_ASSERT(mBordersY);
  MOZ_ASSERT(Slots() > 0);

#ifdef USE_SSE2
  if (mozilla::supports_avx2()) {
    InAvx2(&mState, aInputRow);
    return;
  }
  if (mozilla::supports_sse2()) {
    InSse2(&mState, aInputRow);
    return;
  }
#elif defined(USE_NEON)
  if (mozilla::supports_neon()) {
    InNeon(&mState, aInputRow);
    return;
  }
#endif
  ScaleInputRow(aInputRow);
}

void StreamingScaler::ProduceRow(uint8_t* aOutputRow) {
  MOZ_ASSERT(mBordersY);
  MOZ_ASSERT(Slots() == 0);

#ifdef USE_SSE2
  if (mozilla::supports_avx2()) {
    OutAvx2(&mState, aOutputRow);
    return;
  }
  if (mozilla::supports_sse2()) {
    OutSse2(&mState, aOutputRow);
    return;
  }
#elif defined(USE_NEON)
  if (mozilla::supports_neon()) {
    OutNeon(&mState, aOutputRow);
    return;
  }
#endif
  YScaleOut(mState.mSumsY, mState.mOutWidth, aOutputRow, mState.mHasAlpha,
            mState.mSumsYTap);
  mState.mSumsYTap = (mState.mSumsYTap + 1) & 3;
  mState.mOutPos++;
}

bool StreamingScaler::OutputComplete() const {
  MOZ_ASSERT(mBordersY);
  return mState.mOutPos >= mState.mOutHeight;
}

void StreamingScaler::Reset() {
  if (mBordersY) {
    mState.mInPos = mState.mOutPos = 0;
    mState.mSumsYTap = 0;
    memset(static_cast<float*>(mSumsY), 0,
           sizeof(float) * mState.mOutWidth * 4 * kTaps);
    InitCoefficients();
  }
}


void StreamingScaler::InScalar(State* aOs, const uint8_t* aIn) {
  float* coeffsY = aOs->mCoeffsY + aOs->mInPos * 4;

  if (aOs->mHasAlpha) {
    ScaleDownBgra(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                  aOs->mBordersX, coeffsY, aOs->mSumsYTap);
  } else {
    ScaleDownBgrx(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                  aOs->mBordersX, coeffsY, aOs->mSumsYTap);
  }

  aOs->mBordersY[aOs->mOutPos] -= 1;
  aOs->mInPos++;
}

void StreamingScaler::OutScalar(State* aOs, uint8_t* aOut) {
  YScaleOut(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mHasAlpha,
            aOs->mSumsYTap);
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;
  aOs->mOutPos++;
}

}  // namespace mozilla::gfx
