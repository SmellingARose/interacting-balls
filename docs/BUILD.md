# Build

**macOS** (CPU + Metal + OpenCL)
```bash
cmake -S . -B build && cmake --build build -j
```
App bundle: put `build/brtrain` in `Ball Arena Trainer.app/Contents/MacOS/` with an `Info.plist` (`LSUIElement` true).

**Windows** (CPU + OpenCL), natively with MinGW-w64 GCC and Ninja:
```bash
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=gcc && cmake --build build
```

Or cross-compiled with Zig after a CMake build has generated `build/kernels_embed.c`:
```bash
zig cc -target x86_64-windows-gnu -O3 -ffast-math -mcpu=x86_64_v3 -DBR_HAVE_OPENCL -D_USE_MATH_DEFINES -Isrc \
  src/main.c src/trainer.c src/server.c src/backend_cpu.c src/backend_opencl.c build/kernels_embed.c \
  -lws2_32 -lshell32 -o BallArenaTrainer.exe
```
Needs an AVX2 CPU and the GPU driver's OpenCL.

**Developer options:** `--bench`, `--compare` (CPU vs GPU on one batch), `--train`, `--list-devices`; `--help` lists
every flag, including intercept mode (`--mode tag`, `--atk-range`, `--noise`, `--delay`, …) and `--alt-reward`.
