This is a small C++ dense numerical linear algebra library, with a companion experiments directory 
for evaluating performance.
I mostly follow Trefethen & Bau, "Numerical Linear Algebra" and Golub & Van Loan, "Matrix 
Computations."
The implementation uses `NEON` SIMD on ARM64 systems when available.

### Build

The library is packaged as a C++20 named module (`linalgebra`): 

- CMake 4.1.x
- Ninja
- LLVM Clang ≥ 18 with libc++ (Homebrew LLVM 22 is what's tested; AppleClang
  doesn't yet support module dependency scanning)

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build
```

You can explicitly disable SIMD at configure time with:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DLINEAR_ALGEBRA_SIMD=NONE
```

Valid values are `AUTO` (uses available SIMD) and `NONE` (forces scalar fallback).

### Usage

Import the module:

```cpp
import linalgebra;

int main() {
    linalgebra::Matrix A{{1.0, 2.0}, {3.0, 4.0}};
    linalgebra::Vector b{5.0, 6.0};
    auto lu = linalgebra::lu_factor(A);
    auto x  = linalgebra::lu_solve(lu, b);
}
```

### Run tests

```bash
ctest --test-dir build --output-on-failure
```

### What's implemented

- Matrix / Vector core with SIMD matmul
- Triangular solvers (forward / backward substitution)
- LU factorization with partial pivoting (`lu_factor`, `lu_solve`)
- QR factorization — classical GS, modified GS, and Householder (`qr_classical_gs`, 
  `qr_modified_gs`, `qr_householder`)
- Rank-revealing QR — Householder with column pivoting (`qr_colpiv`); reports numerical rank
  and ensures |R(i,i)| ≥ |R(i+1,i+1)|
- Eigenvalue computation via QR iteration:
  - Unshifted QR (`eigenvalues_unshifted`) — linear convergence, T&B Algorithm 28.1
  - Wilkinson-shifted QR (`eigenvalues_shifted`) — typically cubic convergence, T&B Lecture 29
  - Hessenberg + Givens QR (`eigenvalues_hessenberg`) — O(n²) per step after one O(n³) reduction; 
    ~10–30× faster than `eigenvalues_shifted` for n ≥ 50
  - Francis double-shift QR (`eigenvalues_francis`) — implicit bulge chasing on Hessenberg form;
    handles complex conjugate eigenvalue pairs without complex arithmetic; robust subdiagonal +
    2×2 block deflation with exceptional shifts (GVL §7.5)
- Cholesky factorization (`cholesky_factor`, `cholesky_solve`) 
- Symmetric tridiagonalization
- Eigenvectors via inverse iteration (eigenvectors_inverse_iteration)
- SVD — Golub-Kahan bidiagonalization + QR (svd)
- Conjugate Gradient (solve_cg)
- GMRES (solve_gmres)
- BiCGSTAB (solve_bicgstab)
- Condition number estimation
- Preconditioners (precond_jacobi, precond_ilu0)
- Least squares solver (lstsq)
- Arnoldi iteration (arnoldi)
- Matrix exponential (expm)
