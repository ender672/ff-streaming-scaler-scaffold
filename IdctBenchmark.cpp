#include "MozShims.h"
#include "SkConvolver.h"
#include "ImageIO.h"
#include "JpegIdctScale.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace mozilla::gfx;

// This benchmark compares two JPEG downscale pipelines on the same input:
//
//   1. IDCT + Lanczos3: decode at a reduced resolution (1/2, 1/4, 1/8) via
//      libjpeg-turbo's IDCT scaling (same denom-selection logic as Firefox's
//      nsJPEGDecoder, see JpegIdctScale.h), then Lanczos3 the intermediate
//      to the final target.
//
//   2. Pure Lanczos3: decode the JPEG at full resolution, then Lanczos3
//      directly to the final target.
//
// Each (config, approach) pair runs in a forked child so that peak resident
// set size -- reported via getrusage(RUSAGE_SELF) -- is isolated from other
// runs. The minimum CPU time over --iterations runs is reported, matching
// the existing benchmark's methodology.
//
// The six configs exercise IDCT denominators 1/2, 1/4, 1/8 at an
// idct-min-factor of 2.5 (i.e. the denom is selected only when the
// intermediate is at least 2.5x the target on both axes). For each denom
// there is:
//   - a "barely" case: target sits at the low end of the denom's selection
//     window, so Lanczos3 only has ~2.5x left to downscale.
//   - a "heavy"  case: target sits at the high end, so Lanczos3 is doing
//     nearly the maximum work before the denom would drop.
//
// Target dimensions are hand-picked for giant-map.jpg (4992x4272) so that
// both the width and height ratios clear the intended threshold. If the
// input is not giant-map.jpg the configs will still run, but the intended
// IDCT denom is not guaranteed -- the tool will print what was actually
// chosen.

static double TimeToMs(clock_t aT) {
  return static_cast<double>(aT) * 1000.0 / CLOCKS_PER_SEC;
}

enum class Approach { IdctLanczos, PureLanczos };

struct Config {
  const char* mLabel;
  int mOutW;
  int mOutH;
};

// Configs are tuned for giant-map.jpg (4992x4272) with idct-min-factor 2.5.
// Expected intermediate sizes (from libjpeg-turbo IDCT, which ceil-divides):
//   denom 2 -> 2496x2136, denom 4 -> 1248x1068, denom 8 -> 624x534.
static const Config kConfigs[] = {
    {"1/2 barely",  998, 854},  // Lanczos 2496x2136 ->  998x854, ratio ~2.50
    {"1/2 heavy",   500, 428},  // Lanczos 2496x2136 ->  500x428, ratio ~4.99
    {"1/4 barely",  499, 427},  // Lanczos 1248x1068 ->  499x427, ratio ~2.50
    {"1/4 heavy",   250, 214},  // Lanczos 1248x1068 ->  250x214, ratio ~4.99
    {"1/8 barely",  249, 213},  // Lanczos  624x534  ->  249x213, ratio ~2.51
    {"1/8 heavy",   125, 107},  // Lanczos  624x534  ->  125x107, ratio ~4.99
};
static const int kNumConfigs =
    static_cast<int>(sizeof(kConfigs) / sizeof(kConfigs[0]));

constexpr float kMinFactor = 2.5f;

// End-to-end run: decode + (optional) Lanczos3 convolve. Returns elapsed CPU
// time for the full pipeline.
static clock_t RunOnce(const char* aPath, int aOutW, int aOutH,
                       Approach aApproach, int* aIntermediateW,
                       int* aIntermediateH, int* aIdctDenom) {
  JpegDimensions fullDims = ReadJpegDimensions(aPath);
  int denom = 1;
  if (aApproach == Approach::IdctLanczos) {
    denom = ComputeIdctScaleDenom(fullDims.mWidth, fullDims.mHeight, aOutW,
                                  aOutH, kMinFactor);
  }
  *aIdctDenom = denom;

  clock_t start = clock();

  Image image = LoadJpeg(aPath, denom);
  *aIntermediateW = image.mWidth;
  *aIntermediateH = image.mHeight;

  uint8_t* outBuf = nullptr;
  if (image.mWidth != aOutW || image.mHeight != aOutH) {
    skia::SkLanczosFilter lanczos;
    skia::SkConvolutionFilter1D xFilter;
    if (!xFilter.ComputeFilterValues(lanczos, image.mWidth, aOutW)) {
      fprintf(stderr, "ComputeFilterValues (x) failed\n");
      exit(1);
    }
    skia::SkConvolutionFilter1D yFilter;
    if (!yFilter.ComputeFilterValues(lanczos, image.mHeight, aOutH)) {
      fprintf(stderr, "ComputeFilterValues (y) failed\n");
      exit(1);
    }

    int srcStride = image.mWidth * 4;
    int dstStride = aOutW * 4;
    outBuf = static_cast<uint8_t*>(
        calloc(1, static_cast<size_t>(dstStride) * aOutH));
    if (!outBuf) {
      fprintf(stderr, "Output buffer allocation failed\n");
      exit(1);
    }

    if (!skia::BGRAConvolve2D(image.mBuffer, srcStride,
                              SurfaceFormat::R8G8B8A8, xFilter, yFilter,
                              dstStride, outBuf)) {
      fprintf(stderr, "BGRAConvolve2D failed\n");
      exit(1);
    }
  }

  clock_t end = clock();
  free(image.mBuffer);
  free(outBuf);
  return end - start;
}

struct ChildResult {
  double mMinTimeMs;
  long mPeakRssKb;
  int mIntermediateW;
  int mIntermediateH;
  int mIdctDenom;
};

static bool RunInChild(const char* aPath, const Config& aCfg,
                       Approach aApproach, int aIterations, bool aSilent,
                       ChildResult* aResult) {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    perror("pipe");
    return false;
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    close(pipefd[0]);
    close(pipefd[1]);
    return false;
  }

  if (pid == 0) {
    // Child. Silence stderr chatter from LoadJpeg unless --verbose.
    close(pipefd[0]);
    if (aSilent) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
    }

    clock_t tMin = 0;
    int interW = 0, interH = 0, denom = 1;
    for (int i = 0; i < aIterations; i++) {
      clock_t t = RunOnce(aPath, aCfg.mOutW, aCfg.mOutH, aApproach, &interW,
                          &interH, &denom);
      if (!tMin || t < tMin) tMin = t;
    }

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    long peakKb = ru.ru_maxrss;
#ifdef __APPLE__
    // macOS reports ru_maxrss in bytes; Linux reports KB.
    peakKb /= 1024;
#endif

    ChildResult r{TimeToMs(tMin), peakKb, interW, interH, denom};
    ssize_t n = write(pipefd[1], &r, sizeof(r));
    close(pipefd[1]);
    _exit(n == static_cast<ssize_t>(sizeof(r)) ? 0 : 1);
  }

  close(pipefd[1]);
  ssize_t nread = read(pipefd[0], aResult, sizeof(*aResult));
  close(pipefd[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
      nread != static_cast<ssize_t>(sizeof(*aResult))) {
    fprintf(stderr, "Child failed (status %d, read %zd bytes)\n", status,
            nread);
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  int argPos = 1;
  bool verbose = false;
  while (argPos < argc && argv[argPos][0] == '-') {
    if (strcmp(argv[argPos], "-v") == 0 ||
        strcmp(argv[argPos], "--verbose") == 0) {
      verbose = true;
    } else if (strcmp(argv[argPos], "-h") == 0 ||
               strcmp(argv[argPos], "--help") == 0) {
      fprintf(stderr,
              "Usage: %s [-v] <input.jpg>\n"
              "\n"
              "Compares IDCT-prescale + Lanczos3 against pure Lanczos3\n"
              "across configs that exercise IDCT denominators 1/2, 1/4, 1/8\n"
              "at idct-min-factor 2.5, with two Lanczos pressure levels per\n"
              "denominator ('barely' and 'heavy').\n"
              "\n"
              "ITERATIONS env var sets the number of timing iterations\n"
              "(default 5). The minimum time across iterations is reported.\n"
              "Peak RSS is measured via getrusage in a forked child per run.\n",
              argv[0]);
      return 0;
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[argPos]);
      return 1;
    }
    argPos++;
  }

  if (argc - argPos != 1) {
    fprintf(stderr, "Usage: %s [-v] <input.jpg>\n", argv[0]);
    return 1;
  }

  const char* path = argv[argPos];

  int iterations = 5;
  if (getenv("ITERATIONS")) {
    char* end;
    errno = 0;
    unsigned long ul = strtoul(getenv("ITERATIONS"), &end, 10);
    if (*end != '\0' || errno != 0 || ul == 0 || ul > INT_MAX) {
      fprintf(stderr, "Invalid ITERATIONS env var.\n");
      return 1;
    }
    iterations = static_cast<int>(ul);
  }

  JpegDimensions fullDims = ReadJpegDimensions(path);
  printf("Input: %s (%dx%d), iterations: %d, idct-min-factor: %.2f\n", path,
         fullDims.mWidth, fullDims.mHeight, iterations, kMinFactor);
  printf("\n");

  for (int i = 0; i < kNumConfigs; i++) {
    const Config& c = kConfigs[i];

    ChildResult idctR, pureR;
    if (!RunInChild(path, c, Approach::IdctLanczos, iterations, !verbose,
                    &idctR) ||
        !RunInChild(path, c, Approach::PureLanczos, iterations, !verbose,
                    &pureR)) {
      fprintf(stderr, "Run failed for config %s\n", c.mLabel);
      return 1;
    }

    printf("Config: %-11s  target %4dx%-4d\n", c.mLabel, c.mOutW, c.mOutH);
    printf("  IDCT + Lanczos3   %8.2f ms   %8.2f MB   "
           "[IDCT 1/%d -> %dx%d]\n",
           idctR.mMinTimeMs, idctR.mPeakRssKb / 1024.0, idctR.mIdctDenom,
           idctR.mIntermediateW, idctR.mIntermediateH);
    printf("  pure Lanczos3     %8.2f ms   %8.2f MB\n", pureR.mMinTimeMs,
           pureR.mPeakRssKb / 1024.0);

    double speedup = idctR.mMinTimeMs > 0.0
                         ? pureR.mMinTimeMs / idctR.mMinTimeMs
                         : 0.0;
    double memRatio = idctR.mPeakRssKb > 0
                          ? double(pureR.mPeakRssKb) / double(idctR.mPeakRssKb)
                          : 0.0;
    printf("  ratio (pure/IDCT) %7.2fx      %7.2fx\n", speedup, memRatio);
    printf("\n");
  }

  return 0;
}
