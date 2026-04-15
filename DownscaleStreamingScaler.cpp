#include "MozShims.h"
#include "StreamingScaler.h"
#include "ImageIO.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace mozilla::gfx;

static void PrintUsage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s <input.png|jpg> <output.png> [--width W] [--height H]\n"
          "\n"
          "Downscale a PNG or JPEG image using StreamingScaler.\n"
          "Input format is auto-detected. Specify --width, --height, or\n"
          "both. When only one dimension is given, the other is computed\n"
          "to preserve the aspect ratio. When both are given, the image\n"
          "is scaled to the exact requested dimensions.\n",
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

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
      targetWidth = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
      targetHeight = atoi(argv[++i]);
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

  Image image = LoadImage(inputPath);
  fprintf(stderr, "Loaded %s (%dx%d)\n", inputPath, image.mWidth,
          image.mHeight);

  // Compute output dimensions.
  int outWidth, outHeight;
  if (targetWidth > 0 && targetHeight > 0) {
    outWidth = targetWidth;
    outHeight = targetHeight;
  } else if (targetWidth > 0) {
    double scale = static_cast<double>(targetWidth) / image.mWidth;
    outWidth = targetWidth;
    outHeight = std::max(1, static_cast<int>(lround(image.mHeight * scale)));
  } else {
    double scale = static_cast<double>(targetHeight) / image.mHeight;
    outHeight = targetHeight;
    outWidth = std::max(1, static_cast<int>(lround(image.mWidth * scale)));
  }

  if (outWidth > image.mWidth || outHeight > image.mHeight) {
    fprintf(stderr,
            "Error: output (%dx%d) is larger than input (%dx%d). "
            "StreamingScaler only supports downscaling.\n",
            outWidth, outHeight, image.mWidth, image.mHeight);
    free(image.mBuffer);
    return 1;
  }

  fprintf(stderr, "Scaling to %dx%d\n", outWidth, outHeight);

  StreamingScaler scaler;
  IntSize inSize{image.mWidth, image.mHeight};
  IntSize outSize{outWidth, outHeight};
  if (!scaler.Init(inSize, outSize, SurfaceFormat::R8G8B8A8)) {
    fprintf(stderr, "StreamingScaler::Init failed.\n");
    free(image.mBuffer);
    return 1;
  }

  size_t inRowStride = image.mWidth * 4;
  size_t outRowStride = outWidth * 4;
  auto* outBuffer =
      static_cast<uint8_t*>(malloc(static_cast<size_t>(outHeight) * outRowStride));
  if (!outBuffer) {
    fprintf(stderr, "Unable to allocate output buffer.\n");
    free(image.mBuffer);
    return 1;
  }

  int inLine = 0;
  for (int i = 0; i < outHeight; i++) {
    while (scaler.Slots() > 0) {
      scaler.FeedRow(image.mBuffer + inLine * inRowStride);
      inLine++;
    }
    scaler.ProduceRow(outBuffer + i * outRowStride);
  }

  SavePng(outputPath, outBuffer, outWidth, outHeight, true);
  fprintf(stderr, "Wrote %s\n", outputPath);

  free(image.mBuffer);
  free(outBuffer);
  return 0;
}
