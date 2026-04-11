/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "StreamingScaler.h"

#include <cstring>
#include <immintrin.h>

#include "MozShims.h"

#include "StreamingScalerInternal.h"

namespace mozilla::gfx {

static void YScaleOutBgrxAvx2(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
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
    __m128 v0 = _mm_load_ps(aSums + tapOff);
    __m128 v1 = _mm_load_ps(aSums + 16 + tapOff);
    __m128 v2 = _mm_load_ps(aSums + 32 + tapOff);
    __m128 v3 = _mm_load_ps(aSums + 48 + tapOff);

    v0 = _mm_min_ps(_mm_max_ps(v0, zero), one);
    v1 = _mm_min_ps(_mm_max_ps(v1, zero), one);
    v2 = _mm_min_ps(_mm_max_ps(v2, zero), one);
    v3 = _mm_min_ps(_mm_max_ps(v3, zero), one);

    __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v0, scale), half));
    __m128i i1 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v1, scale), half));
    __m128i i2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v2, scale), half));
    __m128i i3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v3, scale), half));

    i0 = _mm_or_si128(_mm_and_si128(i0, mask), xVal);
    i1 = _mm_or_si128(_mm_and_si128(i1, mask), xVal);
    i2 = _mm_or_si128(_mm_and_si128(i2, mask), xVal);
    i3 = _mm_or_si128(_mm_and_si128(i3, mask), xVal);

    __m128i p01 = _mm_packs_epi32(i0, i1);
    __m128i p23 = _mm_packs_epi32(i2, i3);
    __m128i packed = _mm_packus_epi16(p01, p23);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aOut), packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 32 + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 48 + tapOff), z);

    aSums += 64;
    aOut += 16;
  }

  for (; i < aWidth; i++) {
    __m128 vals = _mm_load_ps(aSums + tapOff);

    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    __m128i idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    idx = _mm_or_si128(_mm_and_si128(idx, mask), xVal);
    __m128i packed = _mm_packs_epi32(idx, idx);
    packed = _mm_packus_epi16(packed, packed);
    *reinterpret_cast<int*>(aOut) = _mm_cvtsi128_si32(packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    aSums += 16;
    aOut += 4;
  }
}

static void ScaleDownBgrxAvx2(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  const float* lut = gI2fMap;

  /* Precompute 256-bit coefficient vectors ordered by physical slot */
  __m256 cyLo, cyHi;
  {
    float cySlot[4];
    for (int k = 0; k < 4; k++) {
      cySlot[k] = aCoeffsYF[(k - aTap + 4) & 3];
    }
    cyLo = _mm256_set_m128(_mm_set1_ps(cySlot[1]), _mm_set1_ps(cySlot[0]));
    cyHi = _mm256_set_m128(_mm_set1_ps(cySlot[3]), _mm_set1_ps(cySlot[2]));
  }

  __m128 sumR = _mm_setzero_ps();
  __m128 sumG = _mm_setzero_ps();
  __m128 sumB = _mm_setzero_ps();

  for (int i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      __m128 sumR2 = _mm_setzero_ps();
      __m128 sumG2 = _mm_setzero_ps();
      __m128 sumB2 = _mm_setzero_ps();

      int j = 0;
      for (; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        __m128 coeffsX = _mm_load_ps(aCoeffsXF);
        __m128 coeffsX2 = _mm_load_ps(aCoeffsXF + 4);

        __m128 sampleX = _mm_set1_ps(lut[px0 & 0xFF]);
        sumR = _mm_fmadd_ps(coeffsX, sampleX, sumR);

        sampleX = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
        sumG = _mm_fmadd_ps(coeffsX, sampleX, sumG);

        sampleX = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
        sumB = _mm_fmadd_ps(coeffsX, sampleX, sumB);

        sampleX = _mm_set1_ps(lut[px1 & 0xFF]);
        sumR2 = _mm_fmadd_ps(coeffsX2, sampleX, sumR2);

        sampleX = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
        sumG2 = _mm_fmadd_ps(coeffsX2, sampleX, sumG2);

        sampleX = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
        sumB2 = _mm_fmadd_ps(coeffsX2, sampleX, sumB2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        __m128 coeffsX = _mm_load_ps(aCoeffsXF);

        __m128 sampleX = _mm_set1_ps(lut[px & 0xFF]);
        sumR = _mm_fmadd_ps(coeffsX, sampleX, sumR);

        sampleX = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
        sumG = _mm_fmadd_ps(coeffsX, sampleX, sumG);

        sampleX = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
        sumB = _mm_fmadd_ps(coeffsX, sampleX, sumB);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = _mm_add_ps(sumR, sumR2);
      sumG = _mm_add_ps(sumG, sumG2);
      sumB = _mm_add_ps(sumB, sumB2);
    } else {
      for (int j = 0; j < aBorderBuf[i]; j++) {
        __m128 coeffsX = _mm_load_ps(aCoeffsXF);

        __m128 sampleX = _mm_set1_ps(lut[aIn[0]]);
        sumR = _mm_fmadd_ps(coeffsX, sampleX, sumR);

        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sumG = _mm_fmadd_ps(coeffsX, sampleX, sumG);

        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sumB = _mm_fmadd_ps(coeffsX, sampleX, sumB);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using 256-bit AVX2 */
    {
      /* Prefetch next pixel's sums_y */
      _mm_prefetch(reinterpret_cast<const char*>(aSumsYOut + 16), _MM_HINT_T0);

      __m128 rg = _mm_unpacklo_ps(sumR, sumG);
      __m128 bx = _mm_unpacklo_ps(sumB, sumB);
      __m128 bgrx = _mm_movelh_ps(rg, bx);

      __m256 bgrx256 = _mm256_set_m128(bgrx, bgrx);

      __m256 sy = _mm256_loadu_ps(aSumsYOut);
      sy = _mm256_fmadd_ps(cyLo, bgrx256, sy);
      _mm256_storeu_ps(aSumsYOut, sy);

      sy = _mm256_loadu_ps(aSumsYOut + 8);
      sy = _mm256_fmadd_ps(cyHi, bgrx256, sy);
      _mm256_storeu_ps(aSumsYOut + 8, sy);

      aSumsYOut += 16;
    }

    sumR = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumR), 4));
    sumG = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumG), 4));
    sumB = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumB), 4));
  }
}

static void YScaleOutBgraAvx2(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
  int tapOff = aTap * 4;
  __m128 scale = _mm_set1_ps(255.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 one = _mm_set1_ps(1.0f);
  __m128 zero = _mm_setzero_ps();
  __m128i z = _mm_setzero_si128();

  int i = 0;
  for (; i + 3 < aWidth; i += 4) {
    /* Pixel 1 */
    __m128 vals = _mm_load_ps(aSums + tapOff);
    __m128 alphaV = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV = _mm_min_ps(_mm_max_ps(alphaV, zero), one);
    if (_mm_cvtss_f32(alphaV) != 0) {
      vals = _mm_mul_ps(vals, _mm_rcp_ps(alphaV));
    }
    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    {
      __m128 hi = _mm_shuffle_ps(vals, alphaV, _MM_SHUFFLE(0, 0, 2, 2));
      vals = _mm_shuffle_ps(vals, hi, _MM_SHUFFLE(2, 0, 1, 0));
    }
    __m128i idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    /* Pixel 2 */
    __m128 vals2 = _mm_load_ps(aSums + 16 + tapOff);
    __m128 alphaV2 = _mm_shuffle_ps(vals2, vals2, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV2 = _mm_min_ps(_mm_max_ps(alphaV2, zero), one);
    if (_mm_cvtss_f32(alphaV2) != 0) {
      vals2 = _mm_mul_ps(vals2, _mm_rcp_ps(alphaV2));
    }
    vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
    {
      __m128 hi2 = _mm_shuffle_ps(vals2, alphaV2, _MM_SHUFFLE(0, 0, 2, 2));
      vals2 = _mm_shuffle_ps(vals2, hi2, _MM_SHUFFLE(2, 0, 1, 0));
    }
    __m128i idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);

    __m128i packed = _mm_packs_epi32(idx, idx2);

    /* Pixel 3 */
    __m128 vals3 = _mm_load_ps(aSums + 32 + tapOff);
    __m128 alphaV3 = _mm_shuffle_ps(vals3, vals3, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV3 = _mm_min_ps(_mm_max_ps(alphaV3, zero), one);
    if (_mm_cvtss_f32(alphaV3) != 0) {
      vals3 = _mm_mul_ps(vals3, _mm_rcp_ps(alphaV3));
    }
    vals3 = _mm_min_ps(_mm_max_ps(vals3, zero), one);
    {
      __m128 hi3 = _mm_shuffle_ps(vals3, alphaV3, _MM_SHUFFLE(0, 0, 2, 2));
      vals3 = _mm_shuffle_ps(vals3, hi3, _MM_SHUFFLE(2, 0, 1, 0));
    }
    __m128i idx3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals3, scale), half));
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 32 + tapOff), z);

    /* Pixel 4 */
    __m128 vals4 = _mm_load_ps(aSums + 48 + tapOff);
    __m128 alphaV4 = _mm_shuffle_ps(vals4, vals4, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV4 = _mm_min_ps(_mm_max_ps(alphaV4, zero), one);
    if (_mm_cvtss_f32(alphaV4) != 0) {
      vals4 = _mm_mul_ps(vals4, _mm_rcp_ps(alphaV4));
    }
    vals4 = _mm_min_ps(_mm_max_ps(vals4, zero), one);
    {
      __m128 hi4 = _mm_shuffle_ps(vals4, alphaV4, _MM_SHUFFLE(0, 0, 2, 2));
      vals4 = _mm_shuffle_ps(vals4, hi4, _MM_SHUFFLE(2, 0, 1, 0));
    }
    __m128i idx4 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals4, scale), half));
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 48 + tapOff), z);

    __m128i packed2 = _mm_packs_epi32(idx3, idx4);
    packed = _mm_packus_epi16(packed, packed2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aOut), packed);

    aSums += 64;
    aOut += 16;
  }

  for (; i < aWidth; i++) {
    __m128 vals = _mm_load_ps(aSums + tapOff);

    __m128 alphaV = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV = _mm_min_ps(_mm_max_ps(alphaV, zero), one);
    if (_mm_cvtss_f32(alphaV) != 0) {
      vals = _mm_mul_ps(vals, _mm_rcp_ps(alphaV));
    }
    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    {
      __m128 hi = _mm_shuffle_ps(vals, alphaV, _MM_SHUFFLE(0, 0, 2, 2));
      vals = _mm_shuffle_ps(vals, hi, _MM_SHUFFLE(2, 0, 1, 0));
    }
    __m128i idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    __m128i packed = _mm_packs_epi32(idx, idx);
    packed = _mm_packus_epi16(packed, packed);
    *reinterpret_cast<int*>(aOut) = _mm_cvtsi128_si32(packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    aSums += 16;
    aOut += 4;
  }
}

static void ScaleDownBgraAvx2(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  const float* lut = gI2fMap;

  __m256 cy256Lo, cy256Hi;
  {
    float cyPhys[4];
    cyPhys[aTap & 3] = aCoeffsYF[0];
    cyPhys[(aTap + 1) & 3] = aCoeffsYF[1];
    cyPhys[(aTap + 2) & 3] = aCoeffsYF[2];
    cyPhys[(aTap + 3) & 3] = aCoeffsYF[3];
    cy256Lo = _mm256_set_m128(_mm_set1_ps(cyPhys[1]), _mm_set1_ps(cyPhys[0]));
    cy256Hi = _mm256_set_m128(_mm_set1_ps(cyPhys[3]), _mm_set1_ps(cyPhys[2]));
  }

  __m128 sumR = _mm_setzero_ps();
  __m128 sumG = _mm_setzero_ps();
  __m128 sumB = _mm_setzero_ps();
  __m128 sumA = _mm_setzero_ps();

  for (int i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      __m128 sumR2 = _mm_setzero_ps();
      __m128 sumG2 = _mm_setzero_ps();
      __m128 sumB2 = _mm_setzero_ps();
      __m128 sumA2 = _mm_setzero_ps();

      int j = 0;
      for (; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        __m128 coeffsX = _mm_load_ps(aCoeffsXF);
        __m128 coeffsX2 = _mm_load_ps(aCoeffsXF + 4);

        __m128 coeffsXA = _mm_mul_ps(coeffsX, _mm_set1_ps(lut[px0 >> 24]));

        __m128 sampleX = _mm_set1_ps(lut[px0 & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumB);

        sumA = _mm_add_ps(coeffsXA, sumA);

        __m128 coeffsX2A = _mm_mul_ps(coeffsX2, _mm_set1_ps(lut[px1 >> 24]));

        sampleX = _mm_set1_ps(lut[px1 & 0xFF]);
        sumR2 = _mm_add_ps(_mm_mul_ps(coeffsX2A, sampleX), sumR2);

        sampleX = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
        sumG2 = _mm_add_ps(_mm_mul_ps(coeffsX2A, sampleX), sumG2);

        sampleX = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
        sumB2 = _mm_add_ps(_mm_mul_ps(coeffsX2A, sampleX), sumB2);

        sumA2 = _mm_add_ps(coeffsX2A, sumA2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        __m128 coeffsX = _mm_load_ps(aCoeffsXF);

        __m128 coeffsXA = _mm_mul_ps(coeffsX, _mm_set1_ps(lut[px >> 24]));

        __m128 sampleX = _mm_set1_ps(lut[px & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumB);

        sumA = _mm_add_ps(coeffsXA, sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = _mm_add_ps(sumR, sumR2);
      sumG = _mm_add_ps(sumG, sumG2);
      sumB = _mm_add_ps(sumB, sumB2);
      sumA = _mm_add_ps(sumA, sumA2);
    } else {
      for (int j = 0; j < aBorderBuf[i]; j++) {
        __m128 coeffsX = _mm_load_ps(aCoeffsXF);

        __m128 coeffsXA = _mm_mul_ps(coeffsX, _mm_set1_ps(lut[aIn[3]]));

        __m128 sampleX = _mm_set1_ps(lut[aIn[0]]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumB);

        sumA = _mm_add_ps(coeffsXA, sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using ring buffer offsets */
    {
      __m128 rg = _mm_unpacklo_ps(sumR, sumG);
      __m128 ba = _mm_unpacklo_ps(sumB, sumA);
      __m128 bgra = _mm_movelh_ps(rg, ba);

      {
        __m256 bgra256 = _mm256_set_m128(bgra, bgra);
        __m256 syLo = _mm256_loadu_ps(aSumsYOut);
        __m256 syHi = _mm256_loadu_ps(aSumsYOut + 8);
        syLo = _mm256_fmadd_ps(cy256Lo, bgra256, syLo);
        syHi = _mm256_fmadd_ps(cy256Hi, bgra256, syHi);
        _mm256_storeu_ps(aSumsYOut, syLo);
        _mm256_storeu_ps(aSumsYOut + 8, syHi);
      }
      aSumsYOut += 16;
    }

    sumR = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumR), 4));
    sumG = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumG), 4));
    sumB = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumB), 4));
    sumA = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumA), 4));
  }
}

/* AVX2 dispatch functions */

static void YScaleOutAvx2(float* aSums, int aWidth, uint8_t* aOut,
                          bool aHasAlpha, int aTap) {
  if (aHasAlpha) {
    YScaleOutBgraAvx2(aSums, aWidth, aOut, aTap);
  } else {
    YScaleOutBgrxAvx2(aSums, aWidth, aOut, aTap);
  }
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
  YScaleOutAvx2(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mHasAlpha,
                aOs->mSumsYTap);
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;
  aOs->mOutPos++;
}

}  // namespace mozilla::gfx
