/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "StreamingScaler.h"

#include <cstring>
#include <immintrin.h>

#include "MozShims.h"

#include "StreamingScalerInternal.h"

namespace mozilla::gfx {

static void YScaleOutBgrxSse2(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
  int tapOff = aTap * 4;
  __m128 scale = _mm_set1_ps(255.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 one = _mm_set1_ps(1.0f);
  __m128 zero = _mm_setzero_ps();
  __m128i z = _mm_setzero_si128();
  __m128i mask = _mm_set_epi32(0, -1, -1, -1);
  __m128i xVal = _mm_set_epi32(255, 0, 0, 0);

  __m128 vals;
  __m128i idx, packed;
  int i = 0;
  for (; i + 1 < aWidth; i += 2) {
    /* Pixel 1: read only the current tap */
    vals = _mm_load_ps(aSums + tapOff);

    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    idx = _mm_or_si128(_mm_and_si128(idx, mask), xVal);

    /* Zero consumed tap */
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    /* Pixel 2 */
    {
      __m128i idx2;
      __m128 vals2;

      vals2 = _mm_load_ps(aSums + 16 + tapOff);

      vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
      idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));
      idx2 = _mm_or_si128(_mm_and_si128(idx2, mask), xVal);

      packed = _mm_packs_epi32(idx, idx2);
      packed = _mm_packus_epi16(packed, packed);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(aOut), packed);

      /* Zero consumed tap */
      _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);
    }

    aSums += 32;
    aOut += 8;
  }

  for (; i < aWidth; i++) {
    vals = _mm_load_ps(aSums + tapOff);

    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    idx = _mm_or_si128(_mm_and_si128(idx, mask), xVal);
    packed = _mm_packs_epi32(idx, idx);
    packed = _mm_packus_epi16(packed, packed);
    *reinterpret_cast<int*>(aOut) = _mm_cvtsi128_si32(packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    aSums += 16;
    aOut += 4;
  }
}

static void ScaleDownBgrxSse2(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  const float* lut = gI2fMap;
  int off0 = aTap * 4;
  int off1 = ((aTap + 1) & 3) * 4;
  int off2 = ((aTap + 2) & 3) * 4;
  int off3 = ((aTap + 3) & 3) * 4;
  __m128 cy0 = _mm_set1_ps(aCoeffsYF[0]);
  __m128 cy1 = _mm_set1_ps(aCoeffsYF[1]);
  __m128 cy2 = _mm_set1_ps(aCoeffsYF[2]);
  __m128 cy3 = _mm_set1_ps(aCoeffsYF[3]);

  __m128 coeffsX, coeffsX2, sampleX;
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

        coeffsX = _mm_load_ps(aCoeffsXF);
        coeffsX2 = _mm_load_ps(aCoeffsXF + 4);

        sampleX = _mm_set1_ps(lut[px0 & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        sampleX = _mm_set1_ps(lut[px1 & 0xFF]);
        sumR2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumR2);

        sampleX = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
        sumG2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumG2);

        sampleX = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
        sumB2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumB2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = _mm_load_ps(aCoeffsXF);

        sampleX = _mm_set1_ps(lut[px & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = _mm_add_ps(sumR, sumR2);
      sumG = _mm_add_ps(sumG, sumG2);
      sumB = _mm_add_ps(sumB, sumB2);
    } else {
      for (int j = 0; j < aBorderBuf[i]; j++) {
        coeffsX = _mm_load_ps(aCoeffsXF);

        sampleX = _mm_set1_ps(lut[aIn[0]]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using ring buffer offsets */
    {
      __m128 rg = _mm_unpacklo_ps(sumR, sumG);
      __m128 bx = _mm_unpacklo_ps(sumB, sumB);
      __m128 bgrx = _mm_movelh_ps(rg, bx);

      __m128 sy;

      sy = _mm_load_ps(aSumsYOut + off0);
      sy = _mm_add_ps(_mm_mul_ps(cy0, bgrx), sy);
      _mm_store_ps(aSumsYOut + off0, sy);

      sy = _mm_load_ps(aSumsYOut + off1);
      sy = _mm_add_ps(_mm_mul_ps(cy1, bgrx), sy);
      _mm_store_ps(aSumsYOut + off1, sy);

      sy = _mm_load_ps(aSumsYOut + off2);
      sy = _mm_add_ps(_mm_mul_ps(cy2, bgrx), sy);
      _mm_store_ps(aSumsYOut + off2, sy);

      sy = _mm_load_ps(aSumsYOut + off3);
      sy = _mm_add_ps(_mm_mul_ps(cy3, bgrx), sy);
      _mm_store_ps(aSumsYOut + off3, sy);

      aSumsYOut += 16;
    }

    sumR = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumR), 4));
    sumG = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumG), 4));
    sumB = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumB), 4));
  }
}

static void YScaleOutBgraSse2(float* aSums, int aWidth, uint8_t* aOut,
                              int aTap) {
  int tapOff = aTap * 4;
  __m128 scale = _mm_set1_ps(255.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 one = _mm_set1_ps(1.0f);
  __m128 zero = _mm_setzero_ps();
  __m128i z = _mm_setzero_si128();

  __m128 vals;
  __m128i idx, packed;
  int i = 0;
  for (; i + 1 < aWidth; i += 2) {
    vals = _mm_load_ps(aSums + tapOff);
    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    {
      __m128i idx2;
      __m128 vals2;

      vals2 = _mm_load_ps(aSums + 16 + tapOff);
      vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
      idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));

      packed = _mm_packs_epi32(idx, idx2);
      packed = _mm_packus_epi16(packed, packed);

      /* Ensure alpha >= max(R,G,B) for premultiplication invariant */
      {
        __m128i s1 = _mm_srli_epi32(packed, 8);
        __m128i mx = _mm_max_epu8(packed, s1);
        __m128i s2 = _mm_srli_epi32(mx, 16);
        mx = _mm_max_epu8(mx, s2);
        packed = _mm_max_epu8(packed, _mm_slli_epi32(mx, 24));
      }

      _mm_storel_epi64(reinterpret_cast<__m128i*>(aOut), packed);

      _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);
    }

    aSums += 32;
    aOut += 8;
  }

  for (; i < aWidth; i++) {
    vals = _mm_load_ps(aSums + tapOff);
    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    packed = _mm_packs_epi32(idx, idx);
    packed = _mm_packus_epi16(packed, packed);

    {
      __m128i s1 = _mm_srli_epi32(packed, 8);
      __m128i mx = _mm_max_epu8(packed, s1);
      __m128i s2 = _mm_srli_epi32(mx, 16);
      mx = _mm_max_epu8(mx, s2);
      packed = _mm_max_epu8(packed, _mm_slli_epi32(mx, 24));
    }

    *reinterpret_cast<int*>(aOut) = _mm_cvtsi128_si32(packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    aSums += 16;
    aOut += 4;
  }
}

static void ScaleDownBgraSse2(const uint8_t* aIn, float* aSumsYOut,
                              int aOutWidth, float* aCoeffsXF, int* aBorderBuf,
                              float* aCoeffsYF, int aTap) {
  int off0 = aTap * 4;
  int off1 = ((aTap + 1) & 3) * 4;
  int off2 = ((aTap + 2) & 3) * 4;
  int off3 = ((aTap + 3) & 3) * 4;
  __m128 cy0 = _mm_set1_ps(aCoeffsYF[0]);
  __m128 cy1 = _mm_set1_ps(aCoeffsYF[1]);
  __m128 cy2 = _mm_set1_ps(aCoeffsYF[2]);
  __m128 cy3 = _mm_set1_ps(aCoeffsYF[3]);

  const float* lut = gI2fMap;

  __m128 coeffsX, coeffsX2, sampleX;
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

        coeffsX = _mm_load_ps(aCoeffsXF);
        coeffsX2 = _mm_load_ps(aCoeffsXF + 4);

        sampleX = _mm_set1_ps(lut[px0 & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        sampleX = _mm_set1_ps(lut[px0 >> 24]);
        sumA = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumA);

        sampleX = _mm_set1_ps(lut[px1 & 0xFF]);
        sumR2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumR2);

        sampleX = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
        sumG2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumG2);

        sampleX = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
        sumB2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumB2);

        sampleX = _mm_set1_ps(lut[px1 >> 24]);
        sumA2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumA2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = _mm_load_ps(aCoeffsXF);

        sampleX = _mm_set1_ps(lut[px & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        sampleX = _mm_set1_ps(lut[px >> 24]);
        sumA = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = _mm_add_ps(sumR, sumR2);
      sumG = _mm_add_ps(sumG, sumG2);
      sumB = _mm_add_ps(sumB, sumB2);
      sumA = _mm_add_ps(sumA, sumA2);
    } else {
      for (int j = 0; j < aBorderBuf[i]; j++) {
        coeffsX = _mm_load_ps(aCoeffsXF);

        sampleX = _mm_set1_ps(lut[aIn[0]]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        sampleX = _mm_set1_ps(lut[aIn[3]]);
        sumA = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumA);

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
        __m128 sy;
        sy = _mm_load_ps(aSumsYOut + off0);
        sy = _mm_add_ps(_mm_mul_ps(cy0, bgra), sy);
        _mm_store_ps(aSumsYOut + off0, sy);

        sy = _mm_load_ps(aSumsYOut + off1);
        sy = _mm_add_ps(_mm_mul_ps(cy1, bgra), sy);
        _mm_store_ps(aSumsYOut + off1, sy);

        sy = _mm_load_ps(aSumsYOut + off2);
        sy = _mm_add_ps(_mm_mul_ps(cy2, bgra), sy);
        _mm_store_ps(aSumsYOut + off2, sy);

        sy = _mm_load_ps(aSumsYOut + off3);
        sy = _mm_add_ps(_mm_mul_ps(cy3, bgra), sy);
        _mm_store_ps(aSumsYOut + off3, sy);
      }
      aSumsYOut += 16;
    }

    sumR = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumR), 4));
    sumG = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumG), 4));
    sumB = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumB), 4));
    sumA = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumA), 4));
  }
}

static void YScaleOutSse2(float* aSums, int aWidth, uint8_t* aOut,
                          bool aHasAlpha, int aTap) {
  if (aHasAlpha) {
    YScaleOutBgraSse2(aSums, aWidth, aOut, aTap);
  } else {
    YScaleOutBgrxSse2(aSums, aWidth, aOut, aTap);
  }
}

void StreamingScaler::InSse2(State* aOs, const uint8_t* aIn) {
  MOZ_ASSERT(aOs->mBordersY[aOs->mOutPos] != 0);
  float* coeffsY = aOs->mCoeffsY + aOs->mInPos * 4;

  if (aOs->mHasAlpha) {
    ScaleDownBgraSse2(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                      aOs->mBordersX, coeffsY, aOs->mSumsYTap);
  } else {
    ScaleDownBgrxSse2(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                      aOs->mBordersX, coeffsY, aOs->mSumsYTap);
  }

  aOs->mBordersY[aOs->mOutPos] -= 1;
  aOs->mInPos++;
}

void StreamingScaler::OutSse2(State* aOs, uint8_t* aOut) {
  MOZ_ASSERT(aOs->mBordersY[aOs->mOutPos] == 0);
  YScaleOutSse2(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mHasAlpha,
                aOs->mSumsYTap);
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;
  aOs->mOutPos++;
}

}  // namespace mozilla::gfx
