#!/bin/bash
# Copies source files from the Firefox tree into the benchmark directory,
# patching includes to use MozShims.h so the benchmark is self-contained.
# After running this, the benchmark/ directory can be tarred up and
# distributed to any Linux or macOS machine with libpng and a C++17 compiler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Path to the Firefox source tree.  Override with:
#   FIREFOX_SRC=/path/to/firefox ./pack.sh
FIREFOX_SRC="${FIREFOX_SRC:-$HOME/temp/firefox}"
GFX_DIR="$FIREFOX_SRC/gfx/2d"

cd "$SCRIPT_DIR"

# Patch function: copy a file, replacing Mozilla-specific includes with shims.
patch_copy() {
    local src="$1"
    local dst="$2"
    sed \
        -e 's|# *include "mozilla/gfx/2D.h"|#include "MozShims.h"|' \
        -e 's|# *include "Tools.h"|#include "MozShims.h"|' \
        -e 's|# *include "mozilla/Vector.h"|#include "MozShims.h"|' \
        -e 's|# *include "mozilla/Attributes.h"|#include "MozShims.h"|' \
        -e 's|# *include "mozilla/Assertions.h"|#include "MozShims.h"|' \
        -e 's|# *include "mozilla/SSE.h"|#include "MozShims.h"|' \
        -e 's|# *include "mozilla/arm.h"|#include "MozShims.h"|' \
        -e 's|# *include "Types.h"|#include "MozShims.h"|' \
        "$src" > "$dst"
    echo "  $dst"
}

echo "Copying StreamingScaler files..."
patch_copy "$GFX_DIR/StreamingScalerInternal.h"  StreamingScalerInternal.h
patch_copy "$GFX_DIR/StreamingScalerSse2.cpp"    StreamingScalerSse2.cpp
patch_copy "$GFX_DIR/StreamingScalerAvx2.cpp"    StreamingScalerAvx2.cpp
patch_copy "$GFX_DIR/StreamingScalerNeon.cpp"    StreamingScalerNeon.cpp

# StreamingScaler.h: patch includes, make internals public for benchmarking,
# add GetState() accessor and InScalar/OutScalar declarations.
echo "  StreamingScaler.h (with benchmark patches)"
sed \
    -e 's|#include "mozilla/gfx/2D.h"|#include "MozShims.h"|' \
    -e 's|#include "Tools.h"|#include "MozShims.h"|' \
    -e 's|^ *private:| public:|' \
    -e '/State mState;/a\
\
  State* GetState() { return \&mState; }\
  static void InScalar(State* aOs, const uint8_t* aIn);\
  static void OutScalar(State* aOs, uint8_t* aOut);' \
    "$GFX_DIR/StreamingScaler.h" > StreamingScaler.h

# StreamingScaler.cpp: patch includes, add InScalar/OutScalar wrappers.
echo "  StreamingScaler.cpp (with benchmark patches)"
sed \
    -e 's|# *include "mozilla/SSE.h"|#include "MozShims.h"|' \
    -e 's|# *include "mozilla/arm.h"|#include "MozShims.h"|' \
    "$GFX_DIR/StreamingScaler.cpp" > StreamingScaler.cpp

# Insert InScalar/OutScalar wrappers before the closing namespace brace.
python3 -c "
import sys
marker = '}  // namespace mozilla::gfx'
insert = '''
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
'''
with open('StreamingScaler.cpp', 'r') as f:
    content = f.read()
content = content.replace(marker, insert + '\n' + marker)
with open('StreamingScaler.cpp', 'w') as f:
    f.write(content)
"

echo "Copying Lanczos3/SkConvolver files..."
patch_copy "$GFX_DIR/ConvolutionFilterSSE2.cpp"  ConvolutionFilterSSE2.cpp
patch_copy "$GFX_DIR/ConvolutionFilterAVX2.cpp"  ConvolutionFilterAVX2.cpp
patch_copy "$GFX_DIR/ConvolutionFilterNEON.cpp"  ConvolutionFilterNEON.cpp

# SkConvolver.h: patch includes, add ConvolveSimdLevel enum and
# BGRAConvolve2DLevel declaration.
echo "  SkConvolver.h (with benchmark patches)"
patch_copy "$GFX_DIR/SkConvolver.h" SkConvolver.h
python3 -c "
marker = '}  // namespace skia'
insert = '''
enum class ConvolveSimdLevel { Scalar, Sse2, Avx2, Neon };

bool BGRAConvolve2DLevel(const unsigned char* sourceData,
                         int sourceByteRowStride,
                         mozilla::gfx::SurfaceFormat format,
                         const SkConvolutionFilter1D& filterX,
                         const SkConvolutionFilter1D& filterY,
                         int outputByteRowStride, unsigned char* output,
                         ConvolveSimdLevel level);
'''
with open('SkConvolver.h', 'r') as f:
    content = f.read()
content = content.replace(marker, insert + '\n' + marker)
with open('SkConvolver.h', 'w') as f:
    f.write(content)
"

# SkConvolver.cpp: patch includes, add BGRAConvolve2DLevel implementation.
echo "  SkConvolver.cpp (with benchmark patches)"
patch_copy "$GFX_DIR/SkConvolver.cpp" SkConvolver.cpp
python3 -c "
marker = '}  // namespace skia'
insert = r'''
static void convolve_horizontally_level(
    const unsigned char* srcData, const SkConvolutionFilter1D& filter,
    unsigned char* outRow, SurfaceFormat format, ConvolveSimdLevel level) {
  bool hasAlpha = !IsOpaque(format);
  switch (level) {
#ifdef USE_SSE2
    case ConvolveSimdLevel::Sse2:
    case ConvolveSimdLevel::Avx2:
      convolve_horizontally_sse2(srcData, filter, outRow, hasAlpha);
      return;
#endif
#ifdef USE_NEON
    case ConvolveSimdLevel::Neon:
      convolve_horizontally_neon(srcData, filter, outRow, hasAlpha);
      return;
#endif
    default:
      break;
  }
  if (hasAlpha) {
    ConvolveHorizontally<true>(srcData, filter, outRow);
  } else {
    ConvolveHorizontally<false>(srcData, filter, outRow);
  }
}

static void convolve_vertically_level(
    const SkConvolutionFilter1D::ConvolutionFixed* filterValues,
    int filterLength, unsigned char* const* sourceDataRows, int pixelWidth,
    unsigned char* outRow, SurfaceFormat format, ConvolveSimdLevel level) {
  bool hasAlpha = !IsOpaque(format);
  switch (level) {
#ifdef USE_SSE2
    case ConvolveSimdLevel::Avx2:
      convolve_vertically_avx2(filterValues, filterLength, sourceDataRows,
                               pixelWidth, outRow, hasAlpha);
      return;
    case ConvolveSimdLevel::Sse2:
      convolve_vertically_sse2(filterValues, filterLength, sourceDataRows,
                               pixelWidth, outRow, hasAlpha);
      return;
#endif
#ifdef USE_NEON
    case ConvolveSimdLevel::Neon:
      convolve_vertically_neon(filterValues, filterLength, sourceDataRows,
                               pixelWidth, outRow, hasAlpha);
      return;
#endif
    default:
      break;
  }
  if (hasAlpha) {
    ConvolveVertically<true>(filterValues, filterLength, sourceDataRows,
                             pixelWidth, outRow);
  } else {
    ConvolveVertically<false>(filterValues, filterLength, sourceDataRows,
                              pixelWidth, outRow);
  }
}

bool BGRAConvolve2DLevel(const unsigned char* sourceData,
                         int sourceByteRowStride, SurfaceFormat format,
                         const SkConvolutionFilter1D& filterX,
                         const SkConvolutionFilter1D& filterY,
                         int outputByteRowStride, unsigned char* output,
                         ConvolveSimdLevel level) {
  int maxYFilterSize = filterY.maxFilter();
  int filterOffset = 0, filterLength = 0;
  const SkConvolutionFilter1D::ConvolutionFixed* filterValues =
      filterY.FilterForValue(0, &filterOffset, &filterLength);
  int nextXRow = filterOffset;

  int rowBufferWidth = (filterX.numValues() + 31) & ~0x1F;
  int rowBufferHeight = maxYFilterSize;

  {
    int64_t size = int64_t(rowBufferWidth) * int64_t(rowBufferHeight);
    if (size > 100 * 1024 * 1024) {
      return false;
    }
  }

  CircularRowBuffer rowBuffer(rowBufferWidth, rowBufferHeight, filterOffset);
  if (!rowBuffer.AllocBuffer()) {
    return false;
  }

  int numOutputRows = filterY.numValues();

  for (int outY = 0; outY < numOutputRows; outY++) {
    filterValues = filterY.FilterForValue(outY, &filterOffset, &filterLength);

    while (nextXRow < filterOffset + filterLength) {
      convolve_horizontally_level(
          &sourceData[(uint64_t)nextXRow * sourceByteRowStride], filterX,
          rowBuffer.advanceRow(), format, level);
      nextXRow++;
    }

    unsigned char* curOutputRow = &output[(uint64_t)outY * outputByteRowStride];

    int firstRowInCircularBuffer;
    unsigned char* const* rowsToConvolve =
        rowBuffer.GetRowAddresses(&firstRowInCircularBuffer);

    unsigned char* const* firstRowForFilter =
        &rowsToConvolve[filterOffset - firstRowInCircularBuffer];

    convolve_vertically_level(filterValues, filterLength, firstRowForFilter,
                              filterX.numValues(), curOutputRow, format, level);
  }
  return true;
}
'''
with open('SkConvolver.cpp', 'r') as f:
    content = f.read()
content = content.replace(marker, insert + '\n' + marker)
with open('SkConvolver.cpp', 'w') as f:
    f.write(content)
"

echo "Done. You can now run 'make' to build the benchmark."
