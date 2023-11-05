#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_Sparse;
use RayBNN_DataLoader;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

use std::collections::HashMap;
use rayon::prelude::*;


#[test]
fn test_remove6() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let neuron_size: u64 = 16;

    let matrix_dims = arrayfire::Dim4::new(&[neuron_size,neuron_size,1,1]);
    let mut W = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/sparse_test.csv",
    );

    //arrayfire::print_gen("W".to_string(), &W, Some(6));

    W = arrayfire::sparse_from_dense(&W, arrayfire::SparseFormat::COO);


    let mut WValues = arrayfire::sparse_get_values(&W);
	let mut WRowIdxCOO = arrayfire::sparse_get_row_indices(&W);
	let mut WColIdx = arrayfire::sparse_get_col_indices(&W);

    let mut modeldata_int: HashMap<String, u64>  = HashMap::new();

    modeldata_int.insert("neuron_size".to_string(), neuron_size.clone());
    modeldata_int.insert("input_size".to_string(), 3);
    modeldata_int.insert("output_size".to_string(), 2);
    modeldata_int.insert("space_dims".to_string(), 3);



    let neuron_idx_dims = arrayfire::Dim4::new(&[1,13,1,1]);
    let mut neuron_idx = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<i32>(
    	"./test_data/neuron_idx.csv",
    );

    neuron_idx = arrayfire::transpose(&neuron_idx, false);



    let glia_pos_dims = arrayfire::Dim4::new(&[2,3,1,1]);
    let mut glia_pos = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/glia_pos.csv",
    );


    let neuron_pos_dims = arrayfire::Dim4::new(&[13,3,1,1]);
    let mut neuron_pos = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/neuron_pos.csv",
    );


    RayBNN_Sparse::Util::Remove::delete_smallest_neurons(
        &modeldata_int,
        &neuron_idx,
        3,
        
    
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );





    let matrix_dims = arrayfire::Dim4::new(&[neuron_size,neuron_size,1,1]);
    let mut W2 = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
        "./test_data/sparse_test4.csv",
    );


    //arrayfire::print_gen("W2".to_string(), &W2, Some(6));


    W2 = arrayfire::sparse_from_dense(&W2, arrayfire::SparseFormat::COO);

    let WValues2 = arrayfire::sparse_get_values(&W2);
	let WRowIdxCOO2 = arrayfire::sparse_get_row_indices(&W2);
	let WColIdx2 = arrayfire::sparse_get_col_indices(&W2);









    let mut WValues_cpu = vec!(f64::default();WValues.elements());
    WValues.host(&mut WValues_cpu);

    let mut WValues2_cpu = vec!(f64::default();WValues2.elements());
    WValues2.host(&mut WValues2_cpu);

    assert_eq!(WValues_cpu, WValues2_cpu);











    let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);



    let mut WRowIdxCOO2_cpu = vec!(i32::default();WRowIdxCOO2.elements());
    WRowIdxCOO2.host(&mut WRowIdxCOO2_cpu);



    assert_eq!(WRowIdxCOO2_cpu, WRowIdxCOO_cpu);






    let mut WColIdx_cpu = vec!(i32::default();WColIdx.elements());
    WColIdx.host(&mut WColIdx_cpu);


    let mut WColIdx2_cpu = vec!(i32::default();WColIdx2.elements());
    WColIdx2.host(&mut WColIdx2_cpu);

    assert_eq!(WColIdx2_cpu, WColIdx_cpu);






}
