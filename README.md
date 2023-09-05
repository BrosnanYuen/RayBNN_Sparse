# RayBNN_Sparse
Sparse Matrix Library for GPUs, CPUs, and FPGAs via CUDA, OpenCL, and oneAPI 

Supports CSR, COO, CSC, and block sparse matrices

Requires Arrayfire and Arrayfire Rust

Supports f16, f32, f64, Complexf16, Complexf32, Complexf64

# Install Arrayfire

Install the Arrayfire 3.9.0 binaries at [https://arrayfire.com/binaries/](https://arrayfire.com/binaries/)

or build from source
[https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire](https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire)


# Add to your Cargo.toml
```
arrayfire = { version = "3.8.1", package = "arrayfire_fork" }
num = "0.4.1"
num-traits = "0.2.16"
half = { version = "2.3.1" , features = ["num-traits"] }
RayBNN_Sparse = "0.1.0"
```



# List of Examples


# Solving a Simple Linear ODE on CUDA with Float 64 bit precision
