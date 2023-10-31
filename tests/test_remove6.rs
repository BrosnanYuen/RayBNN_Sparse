#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_adjacency4() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let neuron_size: u64 = 16;

    let matrix_dims = arrayfire::Dim4::new(&[neuron_size,neuron_size,1,1]);
    let mut W = clusterdiffeq::export::dataloader_f64::file_to_matrix(
    	"./test_data/sparse_test.csv",
    	matrix_dims
    );

    //arrayfire::print_gen("W".to_string(), &W, Some(6));

    W = arrayfire::sparse_from_dense(&W, arrayfire::SparseFormat::COO);


    let mut WValues = arrayfire::sparse_get_values(&W);
	let mut WRowIdxCOO = arrayfire::sparse_get_row_indices(&W);
	let mut WColIdx = arrayfire::sparse_get_col_indices(&W);



	let netdata: clusterdiffeq::neural::network_f64::network_metadata_type = clusterdiffeq::neural::network_f64::network_metadata_type {
		neuron_size: neuron_size,
	    input_size: 3,
		output_size: 2,
		proc_num: 3,
		active_size: neuron_size,
		space_dims: 3,
		step_num: 100,
        batch_size: 1,
		del_unused_neuron: true,

		time_step: 0.1,
		nratio: 0.5,
		neuron_std: 0.3,
		sphere_rad: 0.9,
		neuron_rad: 0.1,
		con_rad: 0.6,
        init_prob: 0.5,
        add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.005,
		spring_const: 0.01,
		repel_const: 10.0
	};

    let neuron_idx_dims = arrayfire::Dim4::new(&[1,13,1,1]);
    let mut neuron_idx = clusterdiffeq::export::dataloader_i32::file_to_matrix(
    	"./test_data/neuron_idx.csv",
    	neuron_idx_dims
    );

    neuron_idx = arrayfire::transpose(&neuron_idx, false);



    let glia_pos_dims = arrayfire::Dim4::new(&[2,3,1,1]);
    let mut glia_pos = clusterdiffeq::export::dataloader_f64::file_to_matrix(
    	"./test_data/glia_pos.csv",
    	glia_pos_dims
    );


    let neuron_pos_dims = arrayfire::Dim4::new(&[13,3,1,1]);
    let mut neuron_pos = clusterdiffeq::export::dataloader_f64::file_to_matrix(
    	"./test_data/neuron_pos.csv",
    	neuron_pos_dims
    );

    
    clusterdiffeq::graph::adjacency_f64::delete_smallest_neurons(
        &netdata,
        &neuron_idx,
        3,
        
    
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
    );





    let matrix_dims = arrayfire::Dim4::new(&[neuron_size,neuron_size,1,1]);
    let mut W2 = clusterdiffeq::export::dataloader_f64::file_to_matrix(
        "./test_data/sparse_test4.csv",
        matrix_dims
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
