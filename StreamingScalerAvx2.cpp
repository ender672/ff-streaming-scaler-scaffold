/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "StreamingScaler.h"

#include <cstring>
#include <immintrin.h>

#include "MozShims.h"

#include "StreamingScalerInternal.h"

namespace mozilla::gfx {

/* Shift smp left by one float lane, zero-filling the top lane. */
static inline __attribute__((always_inline)) __m128 ShiftFLeftAvx2(__m128 v) {
  return (__m128)_mm_srli_si128(_mm_castps_si128(v), 4);
}

/* Pack lane 0 of four vectors into a single vector [a0, b0, c0, d0]. */
static inline __attribute__((always_inline)) __m128 PackLane0X4Avx2(__m128 a,
                                                                   __m128 b,
                                                                   __m128 c,
                                                                   __m128 d) {
  __m128 ab = _mm_unpacklo_ps(a, b);
  __m128 cd = _mm_unpacklo_ps(c, d);
  return _mm_movelh_ps(ab, cd);
}

/* Clamp v to [0,1], multiply by `scale`, round to nearest, and truncate to
 * int32. Produces the byte-range index used by byte packing.
 */
static inline __attribute__((always_inline)) __m128i ClampRoundIdxAvx2(
    __m128 aV, __m128 aZero, __m128 aOne, __m128 aScale, __m128 aHalf) {
  aV = _mm_min_ps(_mm_max_ps(aV, aZero), aOne);
  return _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(aV, aScale), aHalf));
}

/* Build the 256-bit y-coefficient vectors used by YaccFma4Avx2. The
 * y-coefficients are permuted so each physical ring-buffer slot (0..3) receives
 * the matching coefficient for the current tap phase.
 */
static inline __attribute__((always_inline)) void YaccBuildCoeffsAvx2(
    float* aCoeffsYF, int aTap, __m256* aCyLo, __m256* aCyHi) {
  float cy[4];
  cy[aTap & 3] = aCoeffsYF[0];
  cy[(aTap + 1) & 3] = aCoeffsYF[1];
  cy[(aTap + 2) & 3] = aCoeffsYF[2];
  cy[(aTap + 3) & 3] = aCoeffsYF[3];
  *aCyLo = _mm256_set_m128(_mm_set1_ps(cy[1]), _mm_set1_ps(cy[0]));
  *aCyHi = _mm256_set_m128(_mm_set1_ps(cy[3]), _mm_set1_ps(cy[2]));
}

/* Vertical FMA accumulate for a 4-channel output pixel. s0..s3 each hold the
 * horizontal sum for one channel in lane 0; the four lane-0 values are packed
 * into a single 4-float channel vector and FMA'd into the 16-float ring-buffer
 * slice at sums_y_out, using the precomputed per-slot y-coefficients.
 */
static inline __attribute__((always_inline)) void YaccFma4Avx2(
    float* aSumsYOut, __m128 aS0, __m128 aS1, __m128 aS2, __m128 aS3,
    __m256 aCyLo, __m256 aCyHi) {
  __m128 pix = PackLane0X4Avx2(aS0, aS1, aS2, aS3);
  __m256 pix256 = _mm256_set_m128(pix, pix);
  __m256 syLo = _mm256_loadu_ps(aSumsYOut);
  __m256 syHi = _mm256_loadu_ps(aSumsYOut + 8);
  syLo = _mm256_fmadd_ps(aCyLo, pix256, syLo);
  syHi = _mm256_fmadd_ps(aCyHi, pix256, syHi);
  _mm256_storeu_ps(aSumsYOut, syLo);
  _mm256_storeu_ps(aSumsYOut + 8, syHi);
}

/* Per-pixel y-out conversion: clamp, round, and either force X=255 (BGRX) or
 * leave alpha in lane 3 (BGRA).
 */
static inline __attribute__((always_inline)) __m128i YScaleOutIdxAvx2(
    __m128 aVals, __m128 aZero, __m128 aOne, __m128 aScale, __m128 aHalf,
    __m128i aRgbxMask, __m128i aRgbxXVal, int aIsRgbx) {
  __m128i idx = ClampRoundIdxAvx2(aVals, aZero, aOne, aScale, aHalf);
  if (aIsRgbx) {
    return _mm_or_si128(_mm_and_si128(idx, aRgbxMask), aRgbxXVal);
  }
  return idx;
}

/* Enforce alpha >= max(R,G,B) on each packed BGRA pixel. */
static inline __attribute__((always_inline)) __m128i FixupAlphaMaxRgbAvx2(
    __m128i aPacked) {
  __m128i s1 = _mm_srli_epi32(aPacked, 8);
  __m128i mx = _mm_max_epu8(aPacked, s1);
  __m128i s2 = _mm_srli_epi32(mx, 16);
  mx = _mm_max_epu8(mx, s2);
  return _mm_max_epu8(aPacked, _mm_slli_epi32(mx, 24));
}

static inline __attribute__((always_inline)) void YScaleOutAvx2Impl(
    float* aSums, int aWidth, uint8_t* aOut, int aTap, int aIsRgbx) {
  int tapOff = aTap * 4;
  __m128 scale = _mm_set1_ps(255.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 one = _mm_set1_ps(1.0f);
  __m128 zero = _mm_setzero_ps();
  __m128i z = _mm_setzero_si128();
  __m128i mask = _mm_set_epi32(0, -1, -1, -1);
  __m128i xVal = _mm_set_epi32(255, 0, 0, 0);

  int i = 0;
  for (; i + 3 < aWidth; i += 4) {
    __m128i i0 = YScaleOutIdxAvx2(_mm_load_ps(aSums + tapOff), zero, one, scale,
                                  half, mask, xVal, aIsRgbx);
    __m128i i1 = YScaleOutIdxAvx2(_mm_load_ps(aSums + 16 + tapOff), zero, one,
                                  scale, half, mask, xVal, aIsRgbx);
    __m128i i2 = YScaleOutIdxAvx2(_mm_load_ps(aSums + 32 + tapOff), zero, one,
                                  scale, half, mask, xVal, aIsRgbx);
    __m128i i3 = YScaleOutIdxAvx2(_mm_load_ps(aSums + 48 + tapOff), zero, one,
                                  scale, half, mask, xVal, aIsRgbx);

    __m128i p01 = _mm_packs_epi32(i0, i1);
    __m128i p23 = _mm_packs_epi32(i2, i3);
    __m128i packed = _mm_packus_epi16(p01, p23);
    if (!aIsRgbx) {
      packed = FixupAlphaMaxRgbAvx2(packed);
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aOut), packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 32 + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 48 + tapOff), z);

    aSums += 64;
    aOut += 16;
  }

  for (; i < aWidth; i++) {
    __m128i idx = YScaleOutIdxAvx2(_mm_load_ps(aSums + tapOff), zero, one,
                                   scale, half, mask, xVal, aIsRgbx);
    __m128i packed = _mm_packs_epi32(idx, idx);
    packed = _mm_packus_epi16(packed, packed);
    if (!aIsRgbx) {
      packed = FixupAlphaMaxRgbAvx2(packed);
    }
    *reinterpret_cast<int*>(aOut) = _mm_cvtsi128_si32(packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    aSums += 16;
    aOut += 4;
  }
}

static void YScaleOutBgrxAvx2(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
  YScaleOutAvx2Impl(aSums, aWidth, aOut, aTap, 1);
}

static void YScaleOutBgraAvx2(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
  YScaleOutAvx2Impl(aSums, aWidth, aOut, aTap, 0);
}

/* Shared scale_down impl that accumulates `NChannels` channels (3 for BGRX,
 * 4 for BGRA) via per-channel sum vectors, then scatters a 4-channel packed
 * pixel into the tap ring buffer via 256-bit FMA. For BGRX, the X sum (slot 3)
 * is fed from the blue channel since that slot is overwritten to 255 later.
 */
template <int NChannels>
static inline __attribute__((always_inline)) void ScaleDownAvx2Impl(
    const uint8_t* aIn, float* aSumsYOut, int aOutWidth, float* aCoeffsXF,
    int* aBorderBuf, float* aCoeffsYF, int aTap) {
  static_assert(NChannels == 3 || NChannels == 4,
                "StreamingScaler uses BGRX (3) or BGRA (4)");

  const float* lut = gI2fMap.data();

  __m256 cyLo, cyHi;
  YaccBuildCoeffsAvx2(aCoeffsYF, aTap, &cyLo, &cyHi);

  __m128 sum0 = _mm_setzero_ps();
  __m128 sum1 = _mm_setzero_ps();
  __m128 sum2 = _mm_setzero_ps();
  __m128 sum3 = _mm_setzero_ps();

  for (int i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      __m128 sum0b = _mm_setzero_ps();
      __m128 sum1b = _mm_setzero_ps();
      __m128 sum2b = _mm_setzero_ps();
      __m128 sum3b = _mm_setzero_ps();

      int j = 0;
      for (; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        __m128 coeffsX = _mm_load_ps(aCoeffsXF);
        __m128 coeffsX2 = _mm_load_ps(aCoeffsXF + 4);

        __m128 sampleX;

        sampleX = _mm_set1_ps(lut[px0 & 0xFF]);
        sum0 = _mm_fmadd_ps(coeffsX, sampleX, sum0);
        sampleX = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
        sum1 = _mm_fmadd_ps(coeffsX, sampleX, sum1);
        sampleX = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
        sum2 = _mm_fmadd_ps(coeffsX, sampleX, sum2);
        if (NChannels == 4) {
          sampleX = _mm_set1_ps(lut[px0 >> 24]);
          sum3 = _mm_fmadd_ps(coeffsX, sampleX, sum3);
        }

        sampleX = _mm_set1_ps(lut[px1 & 0xFF]);
        sum0b = _mm_fmadd_ps(coeffsX2, sampleX, sum0b);
        sampleX = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
        sum1b = _mm_fmadd_ps(coeffsX2, sampleX, sum1b);
        sampleX = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
        sum2b = _mm_fmadd_ps(coeffsX2, sampleX, sum2b);
        if (NChannels == 4) {
          sampleX = _mm_set1_ps(lut[px1 >> 24]);
          sum3b = _mm_fmadd_ps(coeffsX2, sampleX, sum3b);
        }

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        __m128 coeffsX = _mm_load_ps(aCoeffsXF);
        __m128 sampleX;

        sampleX = _mm_set1_ps(lut[aIn[0]]);
        sum0 = _mm_fmadd_ps(coeffsX, sampleX, sum0);
        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sum1 = _mm_fmadd_ps(coeffsX, sampleX, sum1);
        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sum2 = _mm_fmadd_ps(coeffsX, sampleX, sum2);
        if (NChannels == 4) {
          sampleX = _mm_set1_ps(lut[aIn[3]]);
          sum3 = _mm_fmadd_ps(coeffsX, sampleX, sum3);
        }

        aIn += 4;
        aCoeffsXF += 4;
      }

      sum0 = _mm_add_ps(sum0, sum0b);
      sum1 = _mm_add_ps(sum1, sum1b);
      sum2 = _mm_add_ps(sum2, sum2b);
      if (NChannels == 4) {
        sum3 = _mm_add_ps(sum3, sum3b);
      }
    } else {
      for (int j = 0; j < aBorderBuf[i]; j++) {
        __m128 coeffsX = _mm_load_ps(aCoeffsXF);
        __m128 sampleX;

        sampleX = _mm_set1_ps(lut[aIn[0]]);
        sum0 = _mm_fmadd_ps(coeffsX, sampleX, sum0);
        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sum1 = _mm_fmadd_ps(coeffsX, sampleX, sum1);
        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sum2 = _mm_fmadd_ps(coeffsX, sampleX, sum2);
        if (NChannels == 4) {
          sampleX = _mm_set1_ps(lut[aIn[3]]);
          sum3 = _mm_fmadd_ps(coeffsX, sampleX, sum3);
        }

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* X slot is unused downstream for BGRX; feed sum2 as filler. */
    __m128 fourth = NChannels == 4 ? sum3 : sum2;
    YaccFma4Avx2(aSumsYOut, sum0, sum1, sum2, fourth, cyLo, cyHi);
    aSumsYOut += 16;

    sum0 = ShiftFLeftAvx2(sum0);
    sum1 = ShiftFLeftAvx2(sum1);
    sum2 = ShiftFLeftAvx2(sum2);
    if (NChannels == 4) {
      sum3 = ShiftFLeftAvx2(sum3);
    }
  }
}

static void ScaleDownBgrxAvx2(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  ScaleDownAvx2Impl<3>(aIn, aSumsYOut, aOutWidth, aCoeffsXF, aBorderBuf,
                       aCoeffsYF, aTap);
}

static void ScaleDownBgraAvx2(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  ScaleDownAvx2Impl<4>(aIn, aSumsYOut, aOutWidth, aCoeffsXF, aBorderBuf,
                       aCoeffsYF, aTap);
}

void StreamingScaler::InAvx2(State* aOs, const uint8_t* aIn) {
  MOZ_ASSERT(aOs->mBordersY[aOs->mOutPos] != 0);
  float* coeffsY = aOs->mCoeffsY + aOs->mInPos * 4;

  if (aOs->mHasAlpha) {
    ScaleDownBgraAvx2(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                      aOs->mBordersX, coeffsY, aOs->mSumsYTap);
  } else {
    ScaleDownBgrxAvx2(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                      aOs->mBordersX, coeffsY, aOs->mSumsYTap);
  }

  aOs->mBordersY[aOs->mOutPos] -= 1;
  aOs->mInPos++;
}

void StreamingScaler::OutAvx2(State* aOs, uint8_t* aOut) {
  MOZ_ASSERT(aOs->mBordersY[aOs->mOutPos] == 0);
  if (aOs->mHasAlpha) {
    YScaleOutBgraAvx2(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mSumsYTap);
  } else {
    YScaleOutBgrxAvx2(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mSumsYTap);
  }
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;
  aOs->mOutPos++;
}

}  // namespace mozilla::gfx
