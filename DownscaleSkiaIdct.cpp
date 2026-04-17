#include "MozShims.h"
#include "SkConvolver.h"
#include "ImageIO.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace mozilla::gfx;

static void PrintUsage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s <input.jpg> <output.png> [--width W] [--height H]\n"
          "       [--idct-min-factor N]\n"
          "\n"
          "Input must be JPEG (IDCT pre-scaling requires JPEG decode).\n"
          "\n"
          "Downscale a JPEG image using Firefox's IDCT pre-scaling\n"
          "followed by Skia Lanczos3. The JPEG is first decoded at a\n"
          "reduced resolution (1/2, 1/4, or 1/8) via libjpeg-turbo's\n"
          "IDCT scaling, using the same factor selection as Firefox's\n"
          "nsJPEGDecoder. The intermediate image is then downscaled to\n"
          "the final target size with Skia Lanczos3.\n"
          "\n"
          "--idct-min-factor N  Ensure the IDCT pre-scale produces\n"
          "    an intermediate image at least Nx the target size in\n"
          "    each dimension. The IDCT denominator is backed off until\n"
          "    this is satisfied, so the intermediate will be between\n"
          "    Nx the target and the full original size.\n"
          "    Smaller values force Lanczos3 to do more of the\n"
          "    downscale (slower, but better anti-aliasing).\n"
          "    The IDCT denominator is stepped down through {8,4,2,1}\n"
          "    until the constraint is met, so the actual intermediate\n"
          "    size may be well below the limit when no power-of-2 step\n"
          "    lands in the target range.\n"
          "    Default: no limit (use Firefox's full algorithm).\n"
          "\n"
          "Specify --width, --height, or both.\n",
          argv0);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const char* inputPath = argv[1];
  const char* outputPath = argv[2];
  int targetWidth = 0;
  int targetHeight = 0;
  int idctMinFactor = 0;  // 0 = no limit

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
      targetWidth = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
      targetHeight = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--idct-min-factor") == 0 && i + 1 < argc) {
      idctMinFactor = atoi(argv[++i]);
      if (idctMinFactor < 1) {
        fprintf(stderr, "Error: --idct-min-factor must be >= 1.\n");
        return 1;
      }
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (targetWidth <= 0 && targetHeight <= 0) {
    fprintf(stderr, "Error: specify at least --width or --height.\n");
    PrintUsage(argv[0]);
    return 1;
  }

  // Read JPEG header to get full image dimensions for scale factor computation.
  JpegDimensions fullDims = ReadJpegDimensions(inputPath);
  fprintf(stderr, "Full JPEG dimensions: %dx%d\n", fullDims.mWidth,
          fullDims.mHeight);

  // Compute final output dimensions from full image dimensions (same as the
  // other tools), so the target is identical regardless of IDCT pre-scaling.
  int outWidth, outHeight;
  if (targetWidth > 0 && targetHeight > 0) {
    outWidth = targetWidth;
    outHeight = targetHeight;
  } else if (targetWidth > 0) {
    double scale = static_cast<double>(targetWidth) / fullDims.mWidth;
    outWidth = targetWidth;
    outHeight = std::max(1, static_cast<int>(lround(fullDims.mHeight * scale)));
  } else {
    double scale = static_cast<double>(targetHeight) / fullDims.mHeight;
    outHeight = targetHeight;
    outWidth = std::max(1, static_cast<int>(lround(fullDims.mWidth * scale)));
  }

  if (outWidth > fullDims.mWidth || outHeight > fullDims.mHeight) {
    fprintf(stderr,
            "Error: output (%dx%d) is larger than input (%dx%d). "
            "Only downscaling is supported.\n",
            outWidth, outHeight, fullDims.mWidth, fullDims.mHeight);
    return 1;
  }

  // Compute the IDCT pre-scale factor using Firefox's algorithm.
  int scaleDenom = ComputeIdctScaleDenom(fullDims.mWidth, fullDims.mHeight,
                                         outWidth, outHeight);

  // If --idct-min-factor is set, reduce the denominator until the IDCT output
  // is at least idctMinFactor * target in each dimension.
  if (idctMinFactor > 0) {
    int unclamped = scaleDenom;
    while (scaleDenom > 1) {
      int idctW = (fullDims.mWidth + scaleDenom - 1) / scaleDenom;
      int idctH = (fullDims.mHeight + scaleDenom - 1) / scaleDenom;
      if (idctW >= outWidth * idctMinFactor &&
          idctH >= outHeight * idctMinFactor) {
        break;
      }
      // Step down: 8->4, 4->2, 2->1.
      scaleDenom /= 2;
    }
    fprintf(stderr, "IDCT scale factor: 1/%d (clamped from 1/%d by "
                    "--idct-min-factor %d)\n",
            scaleDenom, unclamped, idctMinFactor);
  } else {
    fprintf(stderr, "IDCT scale factor: 1/%d\n", scaleDenom);
  }

  // Decode the JPEG with IDCT pre-scaling.
  Image image = LoadJpeg(inputPath, scaleDenom);
  fprintf(stderr, "Intermediate size after IDCT: %dx%d\n", image.mWidth,
          image.mHeight);

  // If IDCT scaling already produced the exact target size (or smaller), just
  // write it out directly.
  if (image.mWidth <= outWidth && image.mHeight <= outHeight) {
    fprintf(stderr, "IDCT output already at or below target; no Lanczos pass.\n");
    SavePng(outputPath, image.mBuffer, image.mWidth, image.mHeight, true);
    fprintf(stderr, "Wrote %s\n", outputPath);
    free(image.mBuffer);
    return 0;
  }

  fprintf(stderr, "Scaling %dx%d -> %dx%d with Lanczos3\n", image.mWidth,
          image.mHeight, outWidth, outHeight);

  skia::SkLanczosFilter lanczos;

  skia::SkConvolutionFilter1D xFilter;
  if (!xFilter.ComputeFilterValues(lanczos, image.mWidth, outWidth)) {
    fprintf(stderr, "Failed to compute horizontal filter.\n");
    free(image.mBuffer);
    return 1;
  }

  skia::SkConvolutionFilter1D yFilter;
  if (!yFilter.ComputeFilterValues(lanczos, image.mHeight, outHeight)) {
    fprintf(stderr, "Failed to compute vertical filter.\n");
    free(image.mBuffer);
    return 1;
  }

  int srcStride = image.mWidth * 4;
  int dstStride = outWidth * 4;
  auto* outBuffer =
      static_cast<uint8_t*>(calloc(1, static_cast<size_t>(dstStride) * outHeight));
  if (!outBuffer) {
    fprintf(stderr, "Unable to allocate output buffer.\n");
    free(image.mBuffer);
    return 1;
  }

  if (!skia::BGRAConvolve2D(image.mBuffer, srcStride, SurfaceFormat::R8G8B8A8,
                            xFilter, yFilter, dstStride, outBuffer)) {
    fprintf(stderr, "BGRAConvolve2D failed.\n");
    free(image.mBuffer);
    free(outBuffer);
    return 1;
  }

  SavePng(outputPath, outBuffer, outWidth, outHeight, true);
  fprintf(stderr, "Wrote %s\n", outputPath);

  free(image.mBuffer);
  free(outBuffer);
  return 0;
}
