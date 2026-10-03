# Build

**macOS** (CPU + Metal + OpenCL)
```bash
cmake -S . -B build && cmake --build build -j
```
App bundle: put `build/brtrain` in `Ballistic Range Trainer.app/Contents/MacOS/` with an `Info.plist` (`LSUIElement` true).

**Windows** (CPU + OpenCL), cross-compiled with Zig after a CMake build has generated `build/kernels_embed.c`:
```bash
zig cc -target x86_64-windows-gnu -O3 -ffast-math -mcpu=x86_64_v3 -DBR_HAVE_OPENCL -D_USE_MATH_DEFINES -Isrc \
  src/main.c src/trainer.c src/server.c src/backend_cpu.c src/backend_opencl.c build/kernels_embed.c \
  -lws2_32 -lshell32 -o BallisticRangeTrainer.exe
```
Needs an AVX2 CPU and the GPU driver's OpenCL.

**Developer options:** `--bench`, `--compare`, `--train`, `--list-devices`.
