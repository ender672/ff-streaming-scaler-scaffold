#ifndef JPEG_IDCT_SCALE_H_
#define JPEG_IDCT_SCALE_H_

#include <algorithm>

// IDCT pre-scale denominator selection, excerpted verbatim from Firefox's
// image/decoders/nsJPEGDecoder.cpp (inside the JPEG_START state, around the
// image.jpeg.dct_scaling_enabled check).
//
// Given the full JPEG image dimensions and the final target dimensions,
// returns 1 (no pre-scaling), 2, 4, or 8.
//
// aMinFactor corresponds to Firefox's image.jpeg.dct_scaling_min_factor pref:
// the minimum oversampling ratio (intermediate / target) required before a
// given IDCT denom is picked. 1.0 matches Firefox's default behavior.
inline int ComputeIdctScaleDenom(int aImageWidth, int aImageHeight,
                                 int aTargetWidth, int aTargetHeight,
                                 float aMinFactor = 1.0f) {
  // --- begin verbatim port from nsJPEGDecoder.cpp ---
  // Variables renamed: mInfo.image_width -> aImageWidth, targetSize.width ->
  // aTargetWidth (etc.); writes to mInfo.scale_{num,denom} replaced with
  // `return N`; StaticPrefs::image_jpeg_dct_scaling_min_factor() replaced with
  // the aMinFactor parameter.
  float widthRatio =
      float(aImageWidth) / float(aTargetWidth);
  float heightRatio =
      float(aImageHeight) / float(aTargetHeight);
  float minRatio = std::min(widthRatio, heightRatio);
  float minFactor =
      std::max(1.0f, aMinFactor);

  if (minRatio >= 8.0f * minFactor) {
    return 8;
  } else if (minRatio >= 4.0f * minFactor) {
    return 4;
  } else if (minRatio >= 2.0f * minFactor) {
    return 2;
  }
  // --- end verbatim port ---
  return 1;
}

#endif /* JPEG_IDCT_SCALE_H_ */
