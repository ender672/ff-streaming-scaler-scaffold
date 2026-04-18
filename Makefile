CXX := clang++

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
CXXFLAGS ?= -O3
HOMEBREW_PREFIX := $(shell brew --prefix 2>/dev/null)
ifdef HOMEBREW_PREFIX
CPPFLAGS += -I$(HOMEBREW_PREFIX)/opt/libpng/include
LDFLAGS += -L$(HOMEBREW_PREFIX)/opt/libpng/lib
endif
else
CXXFLAGS ?= -O2
endif

CXXFLAGS += -Wall -std=c++20 -flto=thin
LDFLAGS += -flto=thin

# StreamingScaler objects
SS_OBJS = StreamingScaler.o

# Lanczos3/SkConvolver objects
SK_OBJS = SkConvolver.o

ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
CPPFLAGS += -DUSE_NEON
SS_OBJS += StreamingScalerNeon.o
SK_OBJS += ConvolutionFilterNEON.o
else ifneq ($(filter x86_64,$(shell uname -m)),)
CPPFLAGS += -DUSE_SSE2
SS_OBJS += StreamingScalerSse2.o StreamingScalerAvx2.o
SK_OBJS += ConvolutionFilterSSE2.o ConvolutionFilterAVX2.o
endif

OBJS = $(SS_OBJS) $(SK_OBJS)

all: benchmark idct-benchmark test downscale-streamingscaler downscale-skia downscale-skia-idct

StreamingScaler.o: StreamingScaler.cpp StreamingScaler.h StreamingScalerInternal.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

StreamingScalerSse2.o: StreamingScalerSse2.cpp StreamingScaler.h StreamingScalerInternal.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -msse2 -c -o $@ $<

StreamingScalerAvx2.o: StreamingScalerAvx2.cpp StreamingScaler.h StreamingScalerInternal.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -mavx2 -mfma -c -o $@ $<

StreamingScalerNeon.o: StreamingScalerNeon.cpp StreamingScaler.h StreamingScalerInternal.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

SkConvolver.o: SkConvolver.cpp SkConvolver.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

ConvolutionFilterSSE2.o: ConvolutionFilterSSE2.cpp SkConvolver.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -msse2 -c -o $@ $<

ConvolutionFilterAVX2.o: ConvolutionFilterAVX2.cpp SkConvolver.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -mavx2 -c -o $@ $<

ConvolutionFilterNEON.o: ConvolutionFilterNEON.cpp SkConvolver.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

benchmark: Benchmark.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(OBJS) Benchmark.cpp -o $@ -lpng -lm

test: Test.cpp $(SS_OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(SS_OBJS) Test.cpp -o $@ -lm

downscale-streamingscaler: DownscaleStreamingScaler.cpp $(SS_OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(SS_OBJS) DownscaleStreamingScaler.cpp -o $@ -ljpeg -lpng -lm

downscale-skia: DownscaleSkia.cpp $(SK_OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(SK_OBJS) DownscaleSkia.cpp -o $@ -ljpeg -lpng -lm

downscale-skia-idct: DownscaleSkiaIdct.cpp $(SK_OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(SK_OBJS) DownscaleSkiaIdct.cpp -o $@ -ljpeg -lpng -lm

idct-benchmark: IdctBenchmark.cpp $(SK_OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(SK_OBJS) IdctBenchmark.cpp -o $@ -ljpeg -lpng -lm

clean:
	rm -f benchmark idct-benchmark test downscale-streamingscaler downscale-skia downscale-skia-idct *.o

.PHONY: all clean
