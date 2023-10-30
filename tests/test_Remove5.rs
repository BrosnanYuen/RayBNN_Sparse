
use arrayfire;
use RayBNN_Sparse;
use RayBNN_DataLoader;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

use std::collections::HashMap;
use rayon::prelude::*;

#[test]
fn test_Remove5() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);





    let neuron_size: u64 = 16;

    let matrix_dims = arrayfire::Dim4::new(&[neuron_size,neuron_size,1,1]);
    let mut W = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/sparse_test3.csv",
    );

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
    	"./test_data/neuron_idx3.csv",
    );

    neuron_idx = arrayfire::transpose(&neuron_idx, false);


    let neuron_pos_dims = arrayfire::Dim4::new(&[13,3,1,1]);
    let mut neuron_pos = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/neuron_pos3.csv",
    );


    RayBNN_Sparse::Util::Remove::select_forward_sphere(
        &modeldata_int, 
        &mut WValues, 
        &mut WRowIdxCOO, 
        &mut WColIdx, 
        &neuron_pos, 
        &neuron_idx
    );












    let mut WValues_cpu = vec!(f64::default();WValues.elements());
    WValues.host(&mut WValues_cpu);

    let mut WValues_act:Vec<f64> = vec![ -2.000000 
    ,-7.000000 
    ,-3.000000 
    ,-9.100000 
    ,-2.100000 
    ,-9.000000 
    ,-4.000000 
    ,-7.000000 
    ,-6.000000 
    ,-99.000000 
    ,-75.000000 
    ,-3.600000 
    ,-2.000000 
    ,-50.000000 
    ,-43.000000 
    ,-22.000000 
    ,-0.300000 
    ,-4.100000 
    ,-0.600000 ];

    WValues_cpu = WValues_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    WValues_act = WValues_act.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();


    assert_eq!(WValues_cpu, WValues_act);
















    let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);


    let mut WRowIdxCOO_act:Vec<i32> = vec![
    4 ,
    4 ,
    5 ,
    6 ,
    3 ,
    5 ,
    6 ,
    9 ,
    10 ,
    11 ,
    12 ,
    9 ,
    10 ,
    11 ,
    12 ,
    14 ,
    15 ,
    14 ,
    15 ];


    assert_eq!(WRowIdxCOO_act, WRowIdxCOO_cpu);



















    let mut WColIdx_cpu = vec!(i32::default();WColIdx.elements());
    WColIdx.host(&mut WColIdx_cpu);



    let mut WColIdx_act:Vec<i32> = vec![
        0, 
        1 ,
        1 ,
        1 ,
        2 ,
        2 ,
        2 ,
        3 ,
        3 ,
        3 ,
        3 ,
        6 ,
        6 ,
        6 ,
        6 ,
       10 ,
       10 ,
       11 ,
       11 
     ];

    assert_eq!(WColIdx_cpu, WColIdx_act);

























    let WValues_cpu:Vec<f64> = vec![9.0,-2.8,5.1,-0.3,4.8,4.1, 1.7, -0.9, 0.3, -2.0, 5.0, -1.0, 0.42, -0.2, 0.1, 3.1, 10.0];
	let mut WValues = arrayfire::Array::new(&WValues_cpu, arrayfire::Dim4::new(&[17, 1, 1, 1]));


    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,1,1,1,2,2,3,3,4,4,5,5,5,6,6,6];
	let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[17, 1, 1, 1]));


    let WColIdx_cpu:Vec<i32> =    vec![2,3,4,5,6,0,1,0,1,0,1,2,3,4,2,3,4];
	let mut WColIdx = arrayfire::Array::new(&WColIdx_cpu, arrayfire::Dim4::new(&[17, 1, 1, 1]));



    let delete_cpu:Vec<i32> = vec![1,3,4];
	let mut delete_idx = arrayfire::Array::new(&delete_cpu, arrayfire::Dim4::new(&[delete_cpu.len() as u64, 1, 1, 1]));

    RayBNN_Sparse::Util::Remove::delete_neurons_at_idx(
        &delete_idx, 
        &mut WValues, 
        &mut WRowIdxCOO, 
        &mut WColIdx
    );
    


        
    let mut WValues_act:Vec<f64> = vec![9.0, -0.3, 4.8, 4.1, -0.9, -2.0, -1.0, 0.1];
    let mut WValues_cpu = vec!(f64::default();WValues.elements());

    WValues.host(&mut WValues_cpu);

    WValues_cpu = WValues_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    WValues_act = WValues_act.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    assert_eq!(WValues_cpu, WValues_act);



    

    let mut WRowIdxCOO_act:Vec<i32> = vec![0,1,1,2,3,4,5,6];
    let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());


    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

    assert_eq!(WRowIdxCOO_cpu, WRowIdxCOO_act);

    







    let mut WColIdx_act:Vec<i32> = vec![2,5,6,0,0,0,2,2];
    let mut WColIdx_cpu = vec!(i32::default();WColIdx.elements());


    WColIdx.host(&mut WColIdx_cpu);

    assert_eq!(WColIdx_cpu, WColIdx_act);



}
