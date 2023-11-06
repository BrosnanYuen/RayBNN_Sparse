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
RayBNN_Sparse = "0.1.5"
```



# List of Examples


# Convert COO to CSR Sparse Matrix

```
let mut WRowIdxCSR = RayBNN_Sparse::Util::Convert::COO_to_CSR(&WRowIdxCOO,7);
```

# Convert CSR to COO Sparse Matrix
```
let mut WRowIdxCOO = RayBNN_Sparse::Util::Convert::CSR_to_COO(&WRowIdxCSR);
```

# Search COO Matrix for value
```
let valsel = RayBNN_Sparse::Util::Search::COO_find(&WRowIdxCOO,&idxsel);
```


# Batch Search COO Matrix for value
```
let valsel = RayBNN_Sparse::Util::Search::COO_batch_find(&WRowIdxCOO,&idxsel,4);
```

# Get global index
```
let global_idx = RayBNN_Sparse::Util::Convert::get_global_weight_idx(
    2000, 
    &WRowIdxCOO, 
    &WColIdx
);
```

# Get global index 2
```
let global_idx = RayBNN_Sparse::Util::Convert::get_global_weight_idx2(
    2000, 
    &WRowIdxCOO, 
    &WColIdx
);
```


# Clear inputs to weighted adjancency matrix
```
RayBNN_Sparse::Util::Remove::clear_input::<f32>(
    &mut WValues,
    &mut WRowIdxCOO,
    &mut WColIdx,
    3
);
```


# Clear output of the weighted adjancency matrix
```
RayBNN_Sparse::Util::Remove::clear_output::<f32>(
    &mut WValues,
    &mut WRowIdxCOO,
    &mut WColIdx,
    7-2
);
```


# Clear input to hidden neurons of the weighted adjancency matrix
```
RayBNN_Sparse::Util::Remove::clear_input_to_hidden::<f64>(
    &mut WValues,
    &mut WRowIdxCOO,
    &mut WColIdx,
    3
);
```


# Delete the smallest weights in the weighted adjancency matrix
```
RayBNN_Sparse::Util::Remove::delete_smallest_weights::<f32>(
    &mut WValues,
    &mut WRowIdxCOO,
    &mut WColIdx,
    3
);
```


# Delete the smallest weights with a random probability in the weighted adjancency matrix
```
RayBNN_Sparse::Util::Remove::delete_weights_with_prob::<f64>(
    &mut WValues,
    &mut WRowIdxCOO,
    &mut WColIdx,
    3
);
```



# Remap rows in  weighted adjancency matrix
```
let valsel = RayBNN_Sparse::Util::Convert::remap_rows(&dictionary, &idx,1000);
```


# Block Matrix Multiplication
```
RayBNN_Sparse::Matrix::Block::matmul::<f64>(
	&input_start,
    &input_end,

    &block_start,
    &block_end,


    &input,
    &block
);
```


# Transpose Block Matrix Multiplication
```
RayBNN_Sparse::Matrix::Block::trans_matmul::<f64>(
	&input_start,
    &input_end,

    &block_start,
    &block_end,


    &input,
    &block
);
```






# Parallel lookup of Arrays
```
let result =  RayBNN_Sparse::Util::Search::parallel_lookup(
    0,
    1,

    &idx_arr,
    &test_arr,
);
```
