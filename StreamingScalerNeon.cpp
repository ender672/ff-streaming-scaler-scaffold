/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "StreamingScaler.h"

#include <arm_neon.h>
#include <cstring>

#include "MozShims.h"

#include "StreamingScalerInternal.h"

namespace mozilla::gfx {

/* Shift smp left by one float lane (lane 0 drops, lane 3 gets zero). */
static inline __attribute__((always_inline)) float32x4_t ShiftFLeftNeon(
    float32x4_t v) {
  return vextq_f32(v, vdupq_n_f32(0), 1);
}

/* Gather lane 0 from 4 float32x4_t vectors into [a0, b0, c0, d0]. */
static inline __attribute__((always_inline)) float32x4_t GatherLane0Neon(
    float32x4_t f0, float32x4_t f1, float32x4_t f2, float32x4_t f3) {
  float32x4_t vals = vsetq_lane_f32(vgetq_lane_f32(f0, 0), vdupq_n_f32(0), 0);
  vals = vsetq_lane_f32(vgetq_lane_f32(f1, 0), vals, 1);
  vals = vsetq_lane_f32(vgetq_lane_f32(f2, 0), vals, 2);
  vals = vsetq_lane_f32(vgetq_lane_f32(f3, 0), vals, 3);
  return vals;
}

/* Accumulate `px` into four ring-buffer slots at off0..off3 via FMA with
 * per-slot coefficient vectors cy0..cy3: sums_y_out[offK] += cyK * px.
 */
static inline __attribute__((always_inline)) void ScatterRingFmaNeon(
    float* aSumsYOut, int aOff0, int aOff1, int aOff2, int aOff3,
    float32x4_t aCy0, float32x4_t aCy1, float32x4_t aCy2, float32x4_t aCy3,
    float32x4_t aPx) {
  vst1q_f32(aSumsYOut + aOff0,
            vfmaq_f32(vld1q_f32(aSumsYOut + aOff0), aCy0, aPx));
  vst1q_f32(aSumsYOut + aOff1,
            vfmaq_f32(vld1q_f32(aSumsYOut + aOff1), aCy1, aPx));
  vst1q_f32(aSumsYOut + aOff2,
            vfmaq_f32(vld1q_f32(aSumsYOut + aOff2), aCy2, aPx));
  vst1q_f32(aSumsYOut + aOff3,
            vfmaq_f32(vld1q_f32(aSumsYOut + aOff3), aCy3, aPx));
}

/* Clamp v to [0,1], multiply by `scale`, round to nearest, and truncate to
 * int32. Produces the byte-range index used by byte packing.
 */
static inline __attribute__((always_inline)) int32x4_t ClampRoundIdxNeon(
    float32x4_t aV, float32x4_t aZero, float32x4_t aOne, float32x4_t aScale,
    float32x4_t aHalf) {
  aV = vminq_f32(vmaxq_f32(aV, aZero), aOne);
  return vcvtq_s32_f32(vfmaq_f32(aHalf, aV, aScale));
}

/* Enforce alpha >= max(R,G,B) on each packed BGRA pixel in `result`.
 * Each 32-bit lane carries a separate pixel with A at byte 3.
 */
static inline __attribute__((always_inline)) uint8x16_t FixupAlphaMaxRgbNeon(
    uint8x16_t aResult) {
  uint8x16_t s1 =
      vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(aResult), 8));
  uint8x16_t mx = vmaxq_u8(aResult, s1);
  uint8x16_t s2 = vreinterpretq_u8_u32(vshrq_n_u32(vreinterpretq_u32_u8(mx), 16));
  mx = vmaxq_u8(mx, s2);
  uint8x16_t maxAlpha =
      vreinterpretq_u8_u32(vshlq_n_u32(vreinterpretq_u32_u8(mx), 24));
  return vmaxq_u8(aResult, maxAlpha);
}

static inline __attribute__((always_inline)) void YScaleOutNeonImpl(
    float* aSums, int aWidth, uint8_t* aOut, int aTap, int aIsRgbx) {
  int tapOff = aTap * 4;
  float32x4_t scaleV = vdupq_n_f32(255.0f);
  float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t zero = vdupq_n_f32(0.0f);
  float32x4_t half = vdupq_n_f32(0.5f);
  float32x4_t z = vdupq_n_f32(0.0f);

  static const uint8_t amask[16] = {0, 0, 0, 255, 0, 0, 0, 255,
                                    0, 0, 0, 255, 0, 0, 0, 255};
  uint8x16_t alphaMask = vld1q_u8(amask);

  int i = 0;
  for (; i + 3 < aWidth; i += 4) {
    int32x4_t i0 = ClampRoundIdxNeon(vld1q_f32(aSums + tapOff), zero, one,
                                     scaleV, half);
    int32x4_t i1 = ClampRoundIdxNeon(vld1q_f32(aSums + 16 + tapOff), zero, one,
                                     scaleV, half);
    int32x4_t i2 = ClampRoundIdxNeon(vld1q_f32(aSums + 32 + tapOff), zero, one,
                                     scaleV, half);
    int32x4_t i3 = ClampRoundIdxNeon(vld1q_f32(aSums + 48 + tapOff), zero, one,
                                     scaleV, half);

    int16x4_t h0 = vqmovn_s32(i0);
    int16x4_t h1 = vqmovn_s32(i1);
    int16x8_t h01 = vcombine_s16(h0, h1);
    int16x4_t h2 = vqmovn_s32(i2);
    int16x4_t h3 = vqmovn_s32(i3);
    int16x8_t h23 = vcombine_s16(h2, h3);

    uint8x8_t b01 = vqmovun_s16(h01);
    uint8x8_t b23 = vqmovun_s16(h23);
    uint8x16_t result = vcombine_u8(b01, b23);

    if (aIsRgbx) {
      result = vbslq_u8(alphaMask, vdupq_n_u8(255), result);
    } else {
      result = FixupAlphaMaxRgbNeon(result);
    }

    vst1q_u8(aOut, result);

    vst1q_f32(aSums + tapOff, z);
    vst1q_f32(aSums + 16 + tapOff, z);
    vst1q_f32(aSums + 32 + tapOff, z);
    vst1q_f32(aSums + 48 + tapOff, z);

    aSums += 64;
    aOut += 16;
  }

  for (; i < aWidth; i++) {
    int32x4_t idx = ClampRoundIdxNeon(vld1q_f32(aSums + tapOff), zero, one,
                                      scaleV, half);

    aOut[0] = vgetq_lane_s32(idx, 0);
    aOut[1] = vgetq_lane_s32(idx, 1);
    aOut[2] = vgetq_lane_s32(idx, 2);
    if (aIsRgbx) {
      aOut[3] = 255;
    } else {
      uint8_t alpha = vgetq_lane_s32(idx, 3);
      uint8_t maxRgb = aOut[0];
      if (aOut[1] > maxRgb) maxRgb = aOut[1];
      if (aOut[2] > maxRgb) maxRgb = aOut[2];
      aOut[3] = alpha > maxRgb ? alpha : maxRgb;
    }

    vst1q_f32(aSums + tapOff, z);

    aSums += 16;
    aOut += 4;
  }
}

static void YScaleOutBgrxNeon(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
  YScaleOutNeonImpl(aSums, aWidth, aOut, aTap, 1);
}

static void YScaleOutBgraNeon(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
  YScaleOutNeonImpl(aSums, aWidth, aOut, aTap, 0);
}

/* Shared scale_down impl that accumulates `NChannels` channels (3 for BGRX,
 * 4 for BGRA) via per-channel sum vectors, then scatters a 4-channel packed
 * pixel into the tap ring buffer via FMA. For BGRX, the X sum (slot 3) is
 * fed from the blue channel since that slot is overwritten to 255 later.
 */
template <int NChannels>
static inline __attribute__((always_inline)) void ScaleDownNeonImpl(
    const uint8_t* aIn, float* aSumsYOut, int aOutWidth, float* aCoeffsXF,
    int* aBorderBuf, float* aCoeffsYF, int aTap) {
  static_assert(NChannels == 3 || NChannels == 4,
                "StreamingScaler uses BGRX (3) or BGRA (4)");

  int off0 = aTap * 4;
  int off1 = ((aTap + 1) & 3) * 4;
  int off2 = ((aTap + 2) & 3) * 4;
  int off3 = ((aTap + 3) & 3) * 4;
  float32x4_t cy0 = vdupq_n_f32(aCoeffsYF[0]);
  float32x4_t cy1 = vdupq_n_f32(aCoeffsYF[1]);
  float32x4_t cy2 = vdupq_n_f32(aCoeffsYF[2]);
  float32x4_t cy3 = vdupq_n_f32(aCoeffsYF[3]);

  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);

  for (int i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      float32x4_t sum0b = vdupq_n_f32(0.0f);
      float32x4_t sum1b = vdupq_n_f32(0.0f);
      float32x4_t sum2b = vdupq_n_f32(0.0f);
      float32x4_t sum3b = vdupq_n_f32(0.0f);

      int j = 0;
      for (; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);
        float32x4_t coeffsX2 = vld1q_f32(aCoeffsXF + 4);

        float32x4_t sampleX;

        sampleX = vdupq_n_f32(gI2fMap[px0 & 0xFF]);
        sum0 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum0);
        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 8) & 0xFF]);
        sum1 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum1);
        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 16) & 0xFF]);
        sum2 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum2);
        if (NChannels == 4) {
          sampleX = vdupq_n_f32(gI2fMap[px0 >> 24]);
          sum3 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum3);
        }

        sampleX = vdupq_n_f32(gI2fMap[px1 & 0xFF]);
        sum0b = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sum0b);
        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 8) & 0xFF]);
        sum1b = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sum1b);
        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 16) & 0xFF]);
        sum2b = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sum2b);
        if (NChannels == 4) {
          sampleX = vdupq_n_f32(gI2fMap[px1 >> 24]);
          sum3b = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sum3b);
        }

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);
        float32x4_t sampleX;

        sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sum0 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum0);
        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sum1 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum1);
        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sum2 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum2);
        if (NChannels == 4) {
          sampleX = vdupq_n_f32(gI2fMap[px >> 24]);
          sum3 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum3);
        }

        aIn += 4;
        aCoeffsXF += 4;
      }

      sum0 = vaddq_f32(sum0, sum0b);
      sum1 = vaddq_f32(sum1, sum1b);
      sum2 = vaddq_f32(sum2, sum2b);
      if (NChannels == 4) {
        sum3 = vaddq_f32(sum3, sum3b);
      }
    } else {
      for (int j = 0; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);
        float32x4_t sampleX;

        sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sum0 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum0);
        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sum1 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum1);
        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sum2 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum2);
        if (NChannels == 4) {
          sampleX = vdupq_n_f32(gI2fMap[px >> 24]);
          sum3 = vaddq_f32(vmulq_f32(coeffsX, sampleX), sum3);
        } else {
          /* Keep a fourth independent accumulator live for BGRX so this loop
           * has enough ILP to hide fadd latency; without it, mild-downscale
           * (e.g. 2560->2048) runs ~40% slower on Apple Silicon. The
           * accumulated value lands in the X slot of the packed pixel, which
           * YScaleOutBgrx overwrites to 255 later. */
          sum3 = vaddq_f32(coeffsX, sum3);
        }

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    ScatterRingFmaNeon(aSumsYOut, off0, off1, off2, off3, cy0, cy1, cy2, cy3,
                       GatherLane0Neon(sum0, sum1, sum2, sum3));
    aSumsYOut += 16;

    sum0 = ShiftFLeftNeon(sum0);
    sum1 = ShiftFLeftNeon(sum1);
    sum2 = ShiftFLeftNeon(sum2);
    sum3 = ShiftFLeftNeon(sum3);
  }
}

static void ScaleDownBgrxNeon(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  ScaleDownNeonImpl<3>(aIn, aSumsYOut, aOutWidth, aCoeffsXF, aBorderBuf,
                       aCoeffsYF, aTap);
}

static void ScaleDownBgraNeon(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  ScaleDownNeonImpl<4>(aIn, aSumsYOut, aOutWidth, aCoeffsXF, aBorderBuf,
                       aCoeffsYF, aTap);
}

void StreamingScaler::InNeon(State* aOs, const uint8_t* aIn) {
  MOZ_ASSERT(aOs->mBordersY[aOs->mOutPos] != 0);
  float* coeffsY = aOs->mCoeffsY + aOs->mInPos * 4;

  if (aOs->mHasAlpha) {
    ScaleDownBgraNeon(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                      aOs->mBordersX, coeffsY, aOs->mSumsYTap);
  } else {
    ScaleDownBgrxNeon(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                      aOs->mBordersX, coeffsY, aOs->mSumsYTap);
  }

  aOs->mBordersY[aOs->mOutPos] -= 1;
  aOs->mInPos++;
}

void StreamingScaler::OutNeon(State* aOs, uint8_t* aOut) {
  MOZ_ASSERT(aOs->mBordersY[aOs->mOutPos] == 0);
  if (aOs->mHasAlpha) {
    YScaleOutBgraNeon(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mSumsYTap);
  } else {
    YScaleOutBgrxNeon(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mSumsYTap);
  }
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;
  aOs->mOutPos++;
}

}  // namespace mozilla::gfx
