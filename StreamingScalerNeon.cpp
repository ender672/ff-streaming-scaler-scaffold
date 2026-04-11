/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "StreamingScaler.h"

#include <arm_neon.h>
#include <cstring>

#include "MozShims.h"

#include "StreamingScalerInternal.h"

namespace mozilla::gfx {

static void YScaleOutBgraNeon(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
  int tapOff = aTap * 4;
  float32x4_t scaleV = vdupq_n_f32(255.0f);
  float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t zero = vdupq_n_f32(0.0f);
  float32x4_t half = vdupq_n_f32(0.5f);
  float32x4_t z = vdupq_n_f32(0.0f);

  for (int i = 0; i < aWidth; i++) {
    /* Read [R, G, B, A] from current tap slot */
    float32x4_t vals = vld1q_f32(aSums + tapOff);

    /* Clamp alpha to [0, 1] */
    float alpha = vgetq_lane_f32(vals, 3);
    if (alpha > 1.0f) {
      alpha = 1.0f;
    } else if (alpha < 0.0f) {
      alpha = 0.0f;
    }
    float32x4_t alphaV = vdupq_n_f32(alpha);

    /* Divide RGB by alpha (skip if alpha == 0) */
    if (alpha != 0) {
      vals = vdivq_f32(vals, alphaV);
    }

    /* Clamp RGB to [0, 1], scale to [0, 255], round */
    vals = vminq_f32(vmaxq_f32(vals, zero), one);
    int32x4_t idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scaleV), half));

    aOut[0] = vgetq_lane_s32(idx, 0);
    aOut[1] = vgetq_lane_s32(idx, 1);
    aOut[2] = vgetq_lane_s32(idx, 2);
    aOut[3] = static_cast<int>(alpha * 255.0f + 0.5f);

    /* Zero consumed tap */
    vst1q_f32(aSums + tapOff, z);

    aSums += 16;
    aOut += 4;
  }
}

static void ScaleDownBgraNeon(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  int off0 = aTap * 4;
  int off1 = ((aTap + 1) & 3) * 4;
  int off2 = ((aTap + 2) & 3) * 4;
  int off3 = ((aTap + 3) & 3) * 4;
  float32x4_t cy0 = vdupq_n_f32(aCoeffsYF[0]);
  float32x4_t cy1 = vdupq_n_f32(aCoeffsYF[1]);
  float32x4_t cy2 = vdupq_n_f32(aCoeffsYF[2]);
  float32x4_t cy3 = vdupq_n_f32(aCoeffsYF[3]);

  float32x4_t sumR = vdupq_n_f32(0.0f);
  float32x4_t sumG = vdupq_n_f32(0.0f);
  float32x4_t sumB = vdupq_n_f32(0.0f);
  float32x4_t sumA = vdupq_n_f32(0.0f);

  for (int i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      float32x4_t sumR2 = vdupq_n_f32(0.0f);
      float32x4_t sumG2 = vdupq_n_f32(0.0f);
      float32x4_t sumB2 = vdupq_n_f32(0.0f);
      float32x4_t sumA2 = vdupq_n_f32(0.0f);

      int j = 0;
      for (; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);
        float32x4_t coeffsX2 = vld1q_f32(aCoeffsXF + 4);

        float32x4_t coeffsXA =
            vmulq_f32(coeffsX, vdupq_n_f32(gI2fMap[px0 >> 24]));

        float32x4_t sampleX = vdupq_n_f32(gI2fMap[px0 & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumB);

        sumA = vaddq_f32(coeffsXA, sumA);

        float32x4_t coeffsX2A =
            vmulq_f32(coeffsX2, vdupq_n_f32(gI2fMap[px1 >> 24]));

        sampleX = vdupq_n_f32(gI2fMap[px1 & 0xFF]);
        sumR2 = vaddq_f32(vmulq_f32(coeffsX2A, sampleX), sumR2);

        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 8) & 0xFF]);
        sumG2 = vaddq_f32(vmulq_f32(coeffsX2A, sampleX), sumG2);

        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 16) & 0xFF]);
        sumB2 = vaddq_f32(vmulq_f32(coeffsX2A, sampleX), sumB2);

        sumA2 = vaddq_f32(coeffsX2A, sumA2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);

        float32x4_t coeffsXA =
            vmulq_f32(coeffsX, vdupq_n_f32(gI2fMap[px >> 24]));

        float32x4_t sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumB);

        sumA = vaddq_f32(coeffsXA, sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = vaddq_f32(sumR, sumR2);
      sumG = vaddq_f32(sumG, sumG2);
      sumB = vaddq_f32(sumB, sumB2);
      sumA = vaddq_f32(sumA, sumA2);
    } else {
      for (int j = 0; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);

        float32x4_t coeffsXA =
            vmulq_f32(coeffsX, vdupq_n_f32(gI2fMap[px >> 24]));

        float32x4_t sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumB);

        sumA = vaddq_f32(coeffsXA, sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using ring buffer offsets */
    {
      float32x4_t bgra =
          vsetq_lane_f32(vgetq_lane_f32(sumR, 0), vdupq_n_f32(0), 0);
      bgra = vsetq_lane_f32(vgetq_lane_f32(sumG, 0), bgra, 1);
      bgra = vsetq_lane_f32(vgetq_lane_f32(sumB, 0), bgra, 2);
      bgra = vsetq_lane_f32(vgetq_lane_f32(sumA, 0), bgra, 3);

      float32x4_t sy = vld1q_f32(aSumsYOut + off0);
      sy = vfmaq_f32(sy, cy0, bgra);
      vst1q_f32(aSumsYOut + off0, sy);

      sy = vld1q_f32(aSumsYOut + off1);
      sy = vfmaq_f32(sy, cy1, bgra);
      vst1q_f32(aSumsYOut + off1, sy);

      sy = vld1q_f32(aSumsYOut + off2);
      sy = vfmaq_f32(sy, cy2, bgra);
      vst1q_f32(aSumsYOut + off2, sy);

      sy = vld1q_f32(aSumsYOut + off3);
      sy = vfmaq_f32(sy, cy3, bgra);
      vst1q_f32(aSumsYOut + off3, sy);

      aSumsYOut += 16;
    }

    sumR = vextq_f32(sumR, vdupq_n_f32(0), 1);
    sumG = vextq_f32(sumG, vdupq_n_f32(0), 1);
    sumB = vextq_f32(sumB, vdupq_n_f32(0), 1);
    sumA = vextq_f32(sumA, vdupq_n_f32(0), 1);
  }
}

static void YScaleOutBgrxNeon(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
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
    float32x4_t v0 = vld1q_f32(aSums + tapOff);
    float32x4_t v1 = vld1q_f32(aSums + 16 + tapOff);
    float32x4_t v2 = vld1q_f32(aSums + 32 + tapOff);
    float32x4_t v3 = vld1q_f32(aSums + 48 + tapOff);

    v0 = vminq_f32(vmaxq_f32(v0, zero), one);
    v1 = vminq_f32(vmaxq_f32(v1, zero), one);
    v2 = vminq_f32(vmaxq_f32(v2, zero), one);
    v3 = vminq_f32(vmaxq_f32(v3, zero), one);

    int32x4_t i0 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v0, scaleV), half));
    int32x4_t i1 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v1, scaleV), half));
    int32x4_t i2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v2, scaleV), half));
    int32x4_t i3 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v3, scaleV), half));

    int16x4_t h0 = vqmovn_s32(i0);
    int16x4_t h1 = vqmovn_s32(i1);
    int16x8_t h01 = vcombine_s16(h0, h1);
    int16x4_t h2 = vqmovn_s32(i2);
    int16x4_t h3 = vqmovn_s32(i3);
    int16x8_t h23 = vcombine_s16(h2, h3);

    uint8x8_t b01 = vqmovun_s16(h01);
    uint8x8_t b23 = vqmovun_s16(h23);
    uint8x16_t result = vcombine_u8(b01, b23);

    /* Set alpha lanes to 255 */
    result = vbslq_u8(alphaMask, vdupq_n_u8(255), result);

    vst1q_u8(aOut, result);

    /* Zero consumed taps */
    vst1q_f32(aSums + tapOff, z);
    vst1q_f32(aSums + 16 + tapOff, z);
    vst1q_f32(aSums + 32 + tapOff, z);
    vst1q_f32(aSums + 48 + tapOff, z);

    aSums += 64;
    aOut += 16;
  }

  for (; i < aWidth; i++) {
    float32x4_t vals = vld1q_f32(aSums + tapOff);
    vals = vminq_f32(vmaxq_f32(vals, zero), one);
    int32x4_t idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scaleV), half));

    aOut[0] = vgetq_lane_s32(idx, 0);
    aOut[1] = vgetq_lane_s32(idx, 1);
    aOut[2] = vgetq_lane_s32(idx, 2);
    aOut[3] = 255;

    vst1q_f32(aSums + tapOff, z);

    aSums += 16;
    aOut += 4;
  }
}

static void ScaleDownBgrxNeon(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  int off0 = aTap * 4;
  int off1 = ((aTap + 1) & 3) * 4;
  int off2 = ((aTap + 2) & 3) * 4;
  int off3 = ((aTap + 3) & 3) * 4;
  float32x4_t cy0 = vdupq_n_f32(aCoeffsYF[0]);
  float32x4_t cy1 = vdupq_n_f32(aCoeffsYF[1]);
  float32x4_t cy2 = vdupq_n_f32(aCoeffsYF[2]);
  float32x4_t cy3 = vdupq_n_f32(aCoeffsYF[3]);
  float32x4_t oneV = vdupq_n_f32(1.0f);

  float32x4_t sumR = vdupq_n_f32(0.0f);
  float32x4_t sumG = vdupq_n_f32(0.0f);
  float32x4_t sumB = vdupq_n_f32(0.0f);
  float32x4_t sumX = vdupq_n_f32(0.0f);

  for (int i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      float32x4_t sumR2 = vdupq_n_f32(0.0f);
      float32x4_t sumG2 = vdupq_n_f32(0.0f);
      float32x4_t sumB2 = vdupq_n_f32(0.0f);
      float32x4_t sumX2 = vdupq_n_f32(0.0f);

      int j = 0;
      for (; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);
        float32x4_t coeffsX2 = vld1q_f32(aCoeffsXF + 4);

        float32x4_t sampleX = vdupq_n_f32(gI2fMap[px0 & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumB);

        sumX = vaddq_f32(vmulq_f32(coeffsX, oneV), sumX);

        sampleX = vdupq_n_f32(gI2fMap[px1 & 0xFF]);
        sumR2 = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sumR2);

        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 8) & 0xFF]);
        sumG2 = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sumG2);

        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 16) & 0xFF]);
        sumB2 = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sumB2);

        sumX2 = vaddq_f32(vmulq_f32(coeffsX2, oneV), sumX2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);

        float32x4_t sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumB);

        sumX = vaddq_f32(vmulq_f32(coeffsX, oneV), sumX);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = vaddq_f32(sumR, sumR2);
      sumG = vaddq_f32(sumG, sumG2);
      sumB = vaddq_f32(sumB, sumB2);
      sumX = vaddq_f32(sumX, sumX2);
    } else {
      for (int j = 0; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        float32x4_t coeffsX = vld1q_f32(aCoeffsXF);

        float32x4_t sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumB);

        sumX = vaddq_f32(vmulq_f32(coeffsX, oneV), sumX);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using ring buffer offsets */
    {
      float32x4_t bgrx =
          vsetq_lane_f32(vgetq_lane_f32(sumR, 0), vdupq_n_f32(0), 0);
      bgrx = vsetq_lane_f32(vgetq_lane_f32(sumG, 0), bgrx, 1);
      bgrx = vsetq_lane_f32(vgetq_lane_f32(sumB, 0), bgrx, 2);
      bgrx = vsetq_lane_f32(vgetq_lane_f32(sumX, 0), bgrx, 3);

      float32x4_t sy = vld1q_f32(aSumsYOut + off0);
      sy = vfmaq_f32(sy, cy0, bgrx);
      vst1q_f32(aSumsYOut + off0, sy);

      sy = vld1q_f32(aSumsYOut + off1);
      sy = vfmaq_f32(sy, cy1, bgrx);
      vst1q_f32(aSumsYOut + off1, sy);

      sy = vld1q_f32(aSumsYOut + off2);
      sy = vfmaq_f32(sy, cy2, bgrx);
      vst1q_f32(aSumsYOut + off2, sy);

      sy = vld1q_f32(aSumsYOut + off3);
      sy = vfmaq_f32(sy, cy3, bgrx);
      vst1q_f32(aSumsYOut + off3, sy);

      aSumsYOut += 16;
    }

    sumR = vextq_f32(sumR, vdupq_n_f32(0), 1);
    sumG = vextq_f32(sumG, vdupq_n_f32(0), 1);
    sumB = vextq_f32(sumB, vdupq_n_f32(0), 1);
    sumX = vextq_f32(sumX, vdupq_n_f32(0), 1);
  }
}

/* NEON dispatch functions */

static void YScaleOutNeon(float* aSums, int aWidth, uint8_t* aOut,
                          bool aHasAlpha, int aTap) {
  if (aHasAlpha) {
    YScaleOutBgraNeon(aSums, aWidth, aOut, aTap);
  } else {
    YScaleOutBgrxNeon(aSums, aWidth, aOut, aTap);
  }
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
  YScaleOutNeon(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mHasAlpha,
                aOs->mSumsYTap);
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;
  aOs->mOutPos++;
}

}  // namespace mozilla::gfx
