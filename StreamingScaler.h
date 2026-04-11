/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MOZILLA_GFX_STREAMINGSCALER_H_
#define MOZILLA_GFX_STREAMINGSCALER_H_

#include "MozShims.h"
#include "MozShims.h"

namespace mozilla::gfx {

/**
 * StreamingScaler is a streaming image downscaler.
 */
class StreamingScaler final {
 public:
  StreamingScaler() = default;
  ~StreamingScaler() = default;

  StreamingScaler(const StreamingScaler&) = delete;
  StreamingScaler& operator=(const StreamingScaler&) = delete;

  [[nodiscard]] bool Init(const IntSize& aInputSize, const IntSize& aOutputSize,
                          SurfaceFormat aFormat);

  // Reset for a new progressive pass over the same frame dimensions.
  void Reset();

  // Number of input rows the scaler can accept before producing output.
  [[nodiscard]] int Slots() const;
  // Feed one input row to the scaler.
  void FeedRow(const uint8_t* aInputRow);
  // Produce one output row.
  void ProduceRow(uint8_t* aOutputRow);
  // True when all output rows have been produced.
  [[nodiscard]] bool OutputComplete() const;

 public:
  struct State {
    int32_t mInHeight = 0;
    int32_t mOutHeight = 0;
    int32_t mInWidth = 0;
    int32_t mOutWidth = 0;
    bool mHasAlpha = false;
    int32_t mInPos = 0;
    int32_t mOutPos = 0;
    float* mCoeffsY = nullptr;
    float* mCoeffsX = nullptr;
    int* mBordersX = nullptr;
    int* mBordersY = nullptr;
    float* mSumsY = nullptr;
    float* mTmpCoeffs = nullptr;
    int32_t mSumsYTap = 0;
  };

  void InitCoefficients();
  void ScaleInputRow(const uint8_t* aIn);

#ifdef USE_SSE2
  static void InSse2(State* aOs, const uint8_t* aIn);
  static void OutSse2(State* aOs, uint8_t* aOut);
  static void InAvx2(State* aOs, const uint8_t* aIn);
  static void OutAvx2(State* aOs, uint8_t* aOut);
#endif

#ifdef USE_NEON
  static void InNeon(State* aOs, const uint8_t* aIn);
  static void OutNeon(State* aOs, uint8_t* aOut);
#endif

  State mState;

  State* GetState() { return &mState; }
  static void InScalar(State* aOs, const uint8_t* aIn);
  static void OutScalar(State* aOs, uint8_t* aOut);
  AlignedArray<float> mCoeffsX;
  AlignedArray<int> mBordersX;
  AlignedArray<float> mCoeffsY;
  AlignedArray<int> mBordersY;
  AlignedArray<float> mSumsY;
  AlignedArray<float> mTmpCoeffs;
};

}  // namespace mozilla::gfx

#endif  // MOZILLA_GFX_STREAMINGSCALER_H_
