# Numerical Linear Algebra

This repo contains a small C++ dense numerical linear algebra library for `double`, with a companion experiments directory for evaluating performance.

The current matmul uses a vectorized dot-product kernel. The implementation supports compile-time SIMD backends for `AVX`, `AVX2`, `AVX512`, and `NEON` on `AArch64`/`ARM64` with FP64 vector support.

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

## What's implemented

- Matrix / Vector core with SIMD matmul
- Triangular solvers (forward / backward substitution)
- LU factorization with partial pivoting (`lu_factor`, `lu_solve`)
- QR factorization — classical GS, modified GS, and Householder (`qr_classical_gs`, `qr_modified_gs`, `qr_householder`)

## Run examples

```bash
./build/linear_system
./build/matmul
```

## Run experiments

```bash
./build/pivoting_vs_no_pivoting   
./build/hilbert_qr              
```
