CXXFLAGS ?= -O2
CXXFLAGS += -Wall -std=c++17

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

all: benchmark test

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
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OBJS) Benchmark.cpp -o $@ -lpng -lm

test: Test.cpp $(SS_OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(SS_OBJS) Test.cpp -o $@ -lm

clean:
	rm -f benchmark *.o

.PHONY: all clean
