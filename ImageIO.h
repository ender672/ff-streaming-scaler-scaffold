#ifndef IMAGE_IO_H_
#define IMAGE_IO_H_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "jpeglib.h"
#include "png.h"

struct Image {
  uint8_t* mBuffer;
  int mWidth;
  int mHeight;
};

enum class ImageFormat { Unknown, Png, Jpeg };

// Detect image format by reading the first few bytes of the file.
inline ImageFormat DetectFormat(const char* aPath) {
  FILE* f = fopen(aPath, "rb");
  if (!f) {
    return ImageFormat::Unknown;
  }
  uint8_t header[8];
  size_t n = fread(header, 1, 8, f);
  fclose(f);
  if (n < 4) {
    return ImageFormat::Unknown;
  }
  // JPEG: starts with FF D8 FF
  if (header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF) {
    return ImageFormat::Jpeg;
  }
  // PNG: starts with 89 50 4E 47 0D 0A 1A 0A
  if (n >= 8 && png_sig_cmp(header, 0, 8) == 0) {
    return ImageFormat::Png;
  }
  return ImageFormat::Unknown;
}

inline Image LoadPng(const char* aPath) {
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
    fprintf(stderr, "Input PNG must be RGB or RGBA.\n");
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

// Shared JPEG decompression setup. Sets output to RGBA (JCS_EXT_RGBA).
// If aIdctScaleDenom > 1, sets libjpeg scale_num/scale_denom for IDCT
// pre-scaling before calling jpeg_start_decompress.
inline Image LoadJpeg(const char* aPath, int aIdctScaleDenom = 1) {
  Image image{};

  FILE* input = fopen(aPath, "rb");
  if (!input) {
    fprintf(stderr, "Unable to open input file: %s\n", aPath);
    exit(1);
  }

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, input);
  jpeg_read_header(&cinfo, TRUE);

  // Request RGBA output so all three tools produce identical colorspace data.
  cinfo.out_color_space = JCS_EXT_RGBA;

  // Apply IDCT pre-scaling if requested.
  if (aIdctScaleDenom > 1) {
    cinfo.scale_num = 1;
    cinfo.scale_denom = static_cast<unsigned int>(aIdctScaleDenom);
  }

  jpeg_calc_output_dimensions(&cinfo);
  jpeg_start_decompress(&cinfo);

  image.mWidth = static_cast<int>(cinfo.output_width);
  image.mHeight = static_cast<int>(cinfo.output_height);

  fprintf(stderr, "JPEG %ux%u", cinfo.image_width, cinfo.image_height);
  if (aIdctScaleDenom > 1) {
    fprintf(stderr, " -> IDCT 1/%d -> %dx%d", aIdctScaleDenom, image.mWidth,
            image.mHeight);
  }
  fprintf(stderr, "\n");

  size_t rowStride = static_cast<size_t>(image.mWidth) * 4;
  size_t bufLen = static_cast<size_t>(image.mHeight) * rowStride;
  image.mBuffer = static_cast<uint8_t*>(malloc(bufLen));
  if (!image.mBuffer) {
    fprintf(stderr, "Unable to allocate image buffer.\n");
    exit(1);
  }

  while (cinfo.output_scanline < cinfo.output_height) {
    uint8_t* row = image.mBuffer + cinfo.output_scanline * rowStride;
    jpeg_read_scanlines(&cinfo, &row, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(input);
  return image;
}

// Compute the IDCT pre-scale denominator using the same algorithm as Firefox's
// nsJPEGDecoder (image/decoders/nsJPEGDecoder.cpp, in the StartImage state).
// Returns 1 (no pre-scaling), 2, 4, or 8.
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

// Read JPEG header without decompressing, to get the full image dimensions.
// This is needed by the IDCT tool to compute the scale factor before decoding.
struct JpegDimensions {
  int mWidth;
  int mHeight;
};

inline JpegDimensions ReadJpegDimensions(const char* aPath) {
  FILE* input = fopen(aPath, "rb");
  if (!input) {
    fprintf(stderr, "Unable to open input file: %s\n", aPath);
    exit(1);
  }

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, input);
  jpeg_read_header(&cinfo, TRUE);

  JpegDimensions dims{static_cast<int>(cinfo.image_width),
                      static_cast<int>(cinfo.image_height)};

  jpeg_destroy_decompress(&cinfo);
  fclose(input);
  return dims;
}

inline void SavePng(const char* aPath, const uint8_t* aBuffer, int aWidth,
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

// Load a PNG or JPEG image, auto-detecting the format by file magic.
inline Image LoadImage(const char* aPath) {
  ImageFormat fmt = DetectFormat(aPath);
  if (fmt == ImageFormat::Png) {
    return LoadPng(aPath);
  }
  if (fmt == ImageFormat::Jpeg) {
    return LoadJpeg(aPath);
  }
  fprintf(stderr, "Unrecognized image format: %s\n", aPath);
  exit(1);
}

#endif /* IMAGE_IO_H_ */
