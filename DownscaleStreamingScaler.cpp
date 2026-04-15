#include "MozShims.h"
#include "StreamingScaler.h"

#include "png.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace mozilla::gfx;

struct Image {
  uint8_t* mBuffer;
  int mWidth;
  int mHeight;
};

static Image LoadPng(const char* aPath) {
  Image image{};
  png_structp rpng =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!rpng) {
    fprintf(stderr, "Unable to create PNG read struct.\n");
    exit(1);
  }
  if (setjmp(png_jmpbuf(rpng))) {
    fprintf(stderr, "PNG decoding error.\n");
    exit(1);
  }

  FILE* input = fopen(aPath, "rb");
  if (!input) {
    fprintf(stderr, "Unable to open input file: %s\n", aPath);
    exit(1);
  }

  png_infop rinfo = png_create_info_struct(rpng);
  png_init_io(rpng, input);
  png_read_info(rpng, rinfo);

  int colorType = png_get_color_type(rpng, rinfo);
  if (colorType != PNG_COLOR_TYPE_RGBA && colorType != PNG_COLOR_TYPE_RGB) {
    fprintf(stderr, "Input image must be RGB or RGBA.\n");
    exit(1);
  }

  bool hasAlpha = (colorType == PNG_COLOR_TYPE_RGBA);
  if (!hasAlpha) {
    png_set_filler(rpng, 0xff, PNG_FILLER_AFTER);
  }

  image.mWidth = png_get_image_width(rpng, rinfo);
  image.mHeight = png_get_image_height(rpng, rinfo);

  size_t rowStride = image.mWidth * 4;
  size_t bufLen = static_cast<size_t>(image.mHeight) * rowStride;
  image.mBuffer = static_cast<uint8_t*>(malloc(bufLen));
  auto** rowPtrs =
      static_cast<uint8_t**>(malloc(image.mHeight * sizeof(uint8_t*)));
  if (!image.mBuffer || !rowPtrs) {
    fprintf(stderr, "Unable to allocate buffers.\n");
    exit(1);
  }

  for (int i = 0; i < image.mHeight; i++) {
    rowPtrs[i] = image.mBuffer + i * rowStride;
  }

  png_read_image(rpng, rowPtrs);
  png_destroy_read_struct(&rpng, &rinfo, nullptr);
  free(rowPtrs);
  fclose(input);
  return image;
}

static void SavePng(const char* aPath, const uint8_t* aBuffer, int aWidth,
                    int aHeight, bool aHasAlpha) {
  FILE* output = fopen(aPath, "wb");
  if (!output) {
    fprintf(stderr, "Unable to open output file: %s\n", aPath);
    exit(1);
  }

  png_structp wpng =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!wpng) {
    fprintf(stderr, "Unable to create PNG write struct.\n");
    fclose(output);
    exit(1);
  }
  if (setjmp(png_jmpbuf(wpng))) {
    fprintf(stderr, "PNG encoding error.\n");
    fclose(output);
    exit(1);
  }

  png_infop winfo = png_create_info_struct(wpng);
  png_init_io(wpng, output);

  int colorType = aHasAlpha ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;
  png_set_IHDR(wpng, winfo, aWidth, aHeight, 8, colorType,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  if (!aHasAlpha) {
    png_set_filler(wpng, 0, PNG_FILLER_AFTER);
  }

  png_write_info(wpng, winfo);

  for (int i = 0; i < aHeight; i++) {
    png_write_row(wpng,
                  const_cast<uint8_t*>(aBuffer + i * aWidth * 4));
  }

  png_write_end(wpng, nullptr);
  png_destroy_write_struct(&wpng, &winfo);
  fclose(output);
}

static void PrintUsage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s <input.png> <output.png> [--width W] [--height H]\n"
          "\n"
          "Downscale a PNG image using StreamingScaler while preserving\n"
          "aspect ratio. Specify --width, --height, or both.\n"
          "When only one dimension is given, the other is computed to\n"
          "preserve the aspect ratio. When both are given, the image is\n"
          "scaled to fit within WxH (aspect ratio preserved).\n",
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

  Image image = LoadPng(inputPath);
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
