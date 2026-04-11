# x270 laptop

telliott@fedora:~/temp/ff-streaming-scaler-scaffold$ ./benchmark wp2560.png 
Iterations: 100
2560x1600 StreamingScaler BGRA [scalar]
    to   26x  16  12.43ms
    to  320x 200  17.91ms
    to 2048x1280  67.47ms
2560x1600 StreamingScaler BGRA [sse2]
    to   26x  16   7.16ms
    to  320x 200   9.26ms
    to 2048x1280  23.69ms
2560x1600 StreamingScaler BGRA [avx2]
    to   26x  16   5.20ms
    to  320x 200   6.93ms
    to 2048x1280  17.99ms
2560x1600 lanczos3 BGRA [scalar]
    to   26x  16   7.70ms
    to  320x 200  20.98ms
    to 2048x1280  82.33ms
2560x1600 lanczos3 BGRA [sse2]
    to   26x  16  15.04ms
    to  320x 200  19.82ms
    to 2048x1280  40.09ms
2560x1600 lanczos3 BGRA [avx2]
    to   26x  16  15.12ms
    to  320x 200  18.48ms
    to 2048x1280  31.82ms
2560x1600 StreamingScaler BGRX [scalar]
    to   26x  16   8.71ms
    to  320x 200  13.60ms
    to 2048x1280  51.31ms
2560x1600 StreamingScaler BGRX [sse2]
    to   26x  16   5.82ms
    to  320x 200   7.81ms
    to 2048x1280  19.70ms
2560x1600 StreamingScaler BGRX [avx2]
    to   26x  16   4.37ms
    to  320x 200   5.83ms
    to 2048x1280  13.87ms
2560x1600 lanczos3 BGRX [scalar]
    to   26x  16  19.62ms
    to  320x 200  26.99ms
    to 2048x1280  63.23ms
2560x1600 lanczos3 BGRX [sse2]
    to   26x  16  15.08ms
    to  320x 200  19.84ms
    to 2048x1280  39.78ms
2560x1600 lanczos3 BGRX [avx2]
    to   26x  16  15.08ms
    to  320x 200  18.53ms
    to 2048x1280  31.68ms

# ryzen workstation

# ryzen workstation

telliott@fedora:~/temp/ff-streaming-scaler-scaffold$ ./benchmark wp2560.png 
Iterations: 100
2560x1600 StreamingScaler BGRA [scalar]
    to   26x  16   9.54ms
    to  320x 200  11.91ms
    to 2048x1280  34.31ms
2560x1600 StreamingScaler BGRA [sse2]
    to   26x  16   4.09ms
    to  320x 200   5.50ms
    to 2048x1280  13.14ms
2560x1600 StreamingScaler BGRA [avx2]
    to   26x  16   2.85ms
    to  320x 200   3.86ms
    to 2048x1280  10.62ms
2560x1600 lanczos3 BGRA [scalar]
    to   26x  16   4.76ms
    to  320x 200  13.61ms
    to 2048x1280  62.49ms
2560x1600 lanczos3 BGRA [sse2]
    to   26x  16   6.82ms
    to  320x 200   9.73ms
    to 2048x1280  21.91ms
2560x1600 lanczos3 BGRA [avx2]
    to   26x  16   6.77ms
    to  320x 200   9.05ms
    to 2048x1280  16.96ms
2560x1600 StreamingScaler BGRX [scalar]
    to   26x  16   6.71ms
    to  320x 200   9.29ms
    to 2048x1280  26.63ms
2560x1600 StreamingScaler BGRX [sse2]
    to   26x  16   3.32ms
    to  320x 200   4.46ms
    to 2048x1280  10.66ms
2560x1600 StreamingScaler BGRX [avx2]
    to   26x  16   2.28ms
    to  320x 200   3.09ms
    to 2048x1280   9.03ms
2560x1600 lanczos3 BGRX [scalar]
    to   26x  16  17.09ms
    to  320x 200  21.46ms
    to 2048x1280  46.46ms
2560x1600 lanczos3 BGRX [sse2]
    to   26x  16   6.82ms
    to  320x 200   9.71ms
    to 2048x1280  21.47ms
2560x1600 lanczos3 BGRX [avx2]
    to   26x  16   6.79ms
    to  320x 200   9.10ms
    to 2048x1280  16.81ms

# Macbook air

telliott@Stephanies-MacBook-Air ff-streaming-scaler-scaffold % ./benchmark wp2560.png 
Iterations: 100
2560x1600 StreamingScaler BGRA [scalar]
    to   26x  16   6.26ms
    to  320x 200   6.84ms
    to 2048x1280  14.55ms
2560x1600 StreamingScaler BGRA [neon]
    to   26x  16   2.11ms
    to  320x 200   2.65ms
    to 2048x1280   8.16ms
2560x1600 lanczos3 BGRA [scalar]
    to   26x  16   3.75ms
    to  320x 200   5.70ms
    to 2048x1280  18.96ms
2560x1600 lanczos3 BGRA [neon]
    to   26x  16   9.36ms
    to  320x 200   3.94ms
    to 2048x1280   7.17ms
2560x1600 StreamingScaler BGRX [scalar]
    to   26x  16   5.12ms
    to  320x 200   5.77ms
    to 2048x1280  12.21ms
2560x1600 StreamingScaler BGRX [neon]
    to   26x  16   1.73ms
    to  320x 200   2.14ms
    to 2048x1280   6.60ms
2560x1600 lanczos3 BGRX [scalar]
    to   26x  16   2.99ms
    to  320x 200   5.49ms
    to 2048x1280  25.29ms
2560x1600 lanczos3 BGRX [neon]
    to   26x  16   9.34ms
    to  320x 200   3.95ms
    to 2048x1280   7.14ms



