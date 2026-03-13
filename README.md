# Numerical Linear Algebra

This repo contains a small C++ dense numerical linear algebra library for `double`, with a companion experiments directory for evaluating performance.

The current matmul uses a vectorized dot-product kernel. The implementation supports compile-time SIMD backends for AVX, AVX2, AVX512, NEON (AArch64/ARM64).

## Build

```bash
cmake -S . -B build
cmake --build build
```

On x86, you can explicitly choose a matmul SIMD target at configure time:

```bash
cmake -S . -B build -DLINEAR_ALGEBRA_SIMD=AVX2
```

Valid values are `AUTO`, `NONE`, `AVX`, `AVX2`, and `AVX512`. `AUTO` uses the compiler's current target. `NONE` forces the scalar fallback.

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Run the example

```bash
./build/solve_linear_system
```