# Numerical Linear Algebra

This repo contains a small C++ dense numerical linear algebra library for `double`, with a companion experiments directory for evaluating performance.
I mostly follow Trefethen & Bau, "Numerical Linear Algebra" and Golub & Van Loan, "Matrix Computations" 
The implementation supports compile-time SIMD backends for `AVX`, `AVX2`, `AVX512`, and `NEON` on `AArch64`/`ARM64` with FP64 vector support.

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
- Eigenvalue computation via QR iteration:
  - Unshifted QR (`eigenvalues_unshifted`) — linear convergence, T&B Algorithm 28.1
  - Wilkinson-shifted QR (`eigenvalues_shifted`) — typically cubic convergence, T&B Lecture 29
  - Hessenberg + Givens QR (`eigenvalues_hessenberg`) — O(n²) per step after one O(n³) reduction; ~10–30× faster than `eigenvalues_shifted` for n ≥ 50

TODO:
- [ ] Cholesky factorization (cholesky) — for symmetric positive definite systems
- [ ] Rank-revealing QR — Householder QR with column pivoting (qr_colpiv)
- [ ] Symmetric tridiagonalization — Householder reduction before symmetric QR (tridiagonalize)
- [ ] Francis double-shift QR — bulge chasing for real matrices with complex conjugate eigenvalue pairs (eigenvalues_francis)
- [ ] Deflation — robust subdiagonal + 2×2 block deflation in Hessenberg QR
- [ ] Eigenvectors via inverse iteration (eigenvectors_inverse_iteration)
- [ ] SVD — Golub-Kahan bidiagonalization + QR (svd)
- [ ] Conjugate Gradient (solve_cg) — for symmetric positive definite systems
- [ ] GMRES (solve_gmres) — for general non-symmetric systems
- [ ] BiCGSTAB (solve_bicgstab) — lighter alternative to GMRES
- [ ] Condition number estimation — norm-based LINPACK estimator
- [ ] Preconditioners (precond_jacobi, precond_ilu0) — diagonal and ILU(0) 
- [ ] Least squares solver (lstsq) — via QR or SVD with rank-deficient handling
- [ ] Arnoldi iteration (arnoldi) — falls out naturally from GMRES
- [ ] Matrix exponential (expm) — via Padé approximation

## Run experiments

```bash
./build/matmul
./build/pivoting_vs_no_pivoting   
./build/hilbert_qr              
```
