## MacBook Air, macos

telliott@Stephanies-MacBook-Air ff-streaming-scaler-scaffold % ./benchmark wp2560.png
Iterations: 100
2560x1600 StreamingScaler BGRA [scalar]
    to   26x  16   4.41ms
    to  320x 200   5.08ms
    to 2048x1280  11.50ms
2560x1600 StreamingScaler BGRA [neon]
    to   26x  16   2.03ms
    to  320x 200   2.70ms
    to 2048x1280   8.36ms
2560x1600 lanczos3 BGRA [scalar]
    to   26x  16   3.75ms
    to  320x 200   6.04ms
    to 2048x1280  19.47ms
2560x1600 lanczos3 BGRA [neon]
    to   26x  16   9.38ms
    to  320x 200   3.94ms
    to 2048x1280   7.46ms
2560x1600 StreamingScaler BGRX [scalar]
    to   26x  16   4.04ms
    to  320x 200   4.43ms
    to 2048x1280  10.71ms
2560x1600 StreamingScaler BGRX [neon]
    to   26x  16   1.84ms
    to  320x 200   2.37ms
    to 2048x1280   6.84ms
2560x1600 lanczos3 BGRX [scalar]
    to   26x  16   3.00ms
    to  320x 200   5.57ms
    to 2048x1280  25.46ms
2560x1600 lanczos3 BGRX [neon]
    to   26x  16   9.35ms
    to  320x 200   3.99ms
    to 2048x1280   7.18ms
telliott@Stephanies-MacBook-Air ff-streaming-scaler-scaffold % sysctl -n machdep.cpu.brand_string; echo '---'; sysctl machdep.cpu.core_count machdep.cpu.thread_count; sysctl hw.memsize hw.ncpu
Apple M4
---
machdep.cpu.core_count: 10
machdep.cpu.thread_count: 10
hw.memsize: 17179869184
hw.ncpu: 10

telliott@Stephanies-MacBook-Air ff-streaming-scaler-scaffold % sw_vers
ProductName:        macOS
ProductVersion:     26.4
BuildVersion:       25E246

# Thinkpad x270, Fedora Linux 43

telliott@fedora:~/temp/ff-streaming-scaler-scaffold$ ./benchmark wp2560.png 
Iterations: 100
2560x1600 StreamingScaler BGRA [scalar]
    to   26x  16  12.44ms
    to  320x 200  17.74ms
    to 2048x1280  95.44ms
2560x1600 StreamingScaler BGRA [sse2]
    to   26x  16   7.14ms
    to  320x 200   9.25ms
    to 2048x1280  23.16ms
2560x1600 StreamingScaler BGRA [avx2]
    to   26x  16   5.19ms
    to  320x 200   6.84ms
    to 2048x1280  18.44ms
2560x1600 lanczos3 BGRA [scalar]
    to   26x  16   7.74ms
    to  320x 200  20.82ms
    to 2048x1280  81.95ms
2560x1600 lanczos3 BGRA [sse2]
    to   26x  16  15.03ms
    to  320x 200  19.69ms
    to 2048x1280  39.70ms
2560x1600 lanczos3 BGRA [avx2]
    to   26x  16  14.98ms
    to  320x 200  18.46ms
    to 2048x1280  31.57ms
2560x1600 StreamingScaler BGRX [scalar]
    to   26x  16  11.23ms
    to  320x 200  16.22ms
    to 2048x1280  73.96ms
2560x1600 StreamingScaler BGRX [sse2]
    to   26x  16   6.82ms
    to  320x 200   8.58ms
    to 2048x1280  18.68ms
2560x1600 StreamingScaler BGRX [avx2]
    to   26x  16   4.32ms
    to  320x 200   5.79ms
    to 2048x1280  13.65ms
2560x1600 lanczos3 BGRX [scalar]
    to   26x  16  19.52ms
    to  320x 200  26.81ms
    to 2048x1280  62.79ms
2560x1600 lanczos3 BGRX [sse2]
    to   26x  16  15.02ms
    to  320x 200  19.70ms
    to 2048x1280  39.42ms
2560x1600 lanczos3 BGRX [avx2]
    to   26x  16  14.98ms
    to  320x 200  18.45ms
    to 2048x1280  31.14ms
telliott@fedora:~/temp/ff-streaming-scaler-scaffold$ cat /proc/cpuinfo 
processor   : 0
vendor_id   : GenuineIntel
cpu family  : 6
model       : 142
model name  : Intel(R) Core(TM) i7-7600U CPU @ 2.80GHz
stepping    : 9
microcode   : 0xf6
cpu MHz     : 3899.128
cache size  : 4096 KB

# AMD Ryzen 7 5700X workstation

telliott@fedora:~/temp/ff-streaming-scaler-scaffold$ ./benchmark ../liboil/wp2560.png 
Iterations: 100
2560x1600 StreamingScaler BGRA [scalar]
    to   26x  16  10.87ms
    to  320x 200  13.84ms
    to 2048x1280  56.42ms
2560x1600 StreamingScaler BGRA [sse2]
    to   26x  16   4.06ms
    to  320x 200   5.45ms
    to 2048x1280  13.16ms
2560x1600 StreamingScaler BGRA [avx2]
    to   26x  16   2.86ms
    to  320x 200   3.94ms
    to 2048x1280  10.65ms
2560x1600 lanczos3 BGRA [scalar]
    to   26x  16   4.76ms
    to  320x 200  13.61ms
    to 2048x1280  62.57ms
2560x1600 lanczos3 BGRA [sse2]
    to   26x  16   6.83ms
    to  320x 200   9.74ms
    to 2048x1280  21.90ms
2560x1600 lanczos3 BGRA [avx2]
    to   26x  16   6.78ms
    to  320x 200   9.06ms
    to 2048x1280  16.93ms
2560x1600 StreamingScaler BGRX [scalar]
    to   26x  16  10.84ms
    to  320x 200  13.48ms
    to 2048x1280  49.37ms
2560x1600 StreamingScaler BGRX [sse2]
    to   26x  16   3.34ms
    to  320x 200   4.46ms
    to 2048x1280  10.72ms
2560x1600 StreamingScaler BGRX [avx2]
    to   26x  16   2.40ms
    to  320x 200   3.17ms
    to 2048x1280   9.02ms
2560x1600 lanczos3 BGRX [scalar]
    to   26x  16  17.09ms
    to  320x 200  21.46ms
    to 2048x1280  46.49ms
2560x1600 lanczos3 BGRX [sse2]
    to   26x  16   6.81ms
    to  320x 200   9.72ms
    to 2048x1280  21.47ms
2560x1600 lanczos3 BGRX [avx2]
    to   26x  16   6.78ms
    to  320x 200   9.05ms
    to 2048x1280  16.78ms
telliott@fedora:~/temp/ff-streaming-scaler-scaffold$ cat /proc/cpuinfo 
processor   : 0
vendor_id   : AuthenticAMD
cpu family  : 25
model       : 33
model name  : AMD Ryzen 7 5700X 8-Core Processor
stepping    : 2
microcode   : 0xa201213
cpu MHz     : 3736.149
cache size  : 512 KB

