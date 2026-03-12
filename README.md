# Numerical Linear Algebra

This repo contains a small C++ dense numerical linear algebra library for `double`, with a companion experiments directory for evaluating performance.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Run the example

```bash
./build/solve_linear_system
```