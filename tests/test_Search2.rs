
use arrayfire;
use RayBNN_Sparse;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_search2() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);






    let idx_cpu: [i32; 4] = [222,421,431,652];
	let mut idx = arrayfire::Array::new(&idx_cpu, arrayfire::Dim4::new(&[4, 1, 1, 1]));

    let dictionary_cpu: [i32; 11] = [222,222,222,421,431,431,431,431,431,652,652];
	let mut dictionary = arrayfire::Array::new(&dictionary_cpu, arrayfire::Dim4::new(&[11, 1, 1, 1]));

    
    let valsel = RayBNN_Sparse::Util::Convert::remap_rows(&dictionary, &idx,1000);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![0,0,0,1,2,2,2,2,2,3,3];

    assert_eq!(valsel_act, valsel_cpu);



















    let idx_cpu:Vec<i32> = vec![134,213,216,321,326,431,553];
	let mut idx = arrayfire::Array::new(&idx_cpu, arrayfire::Dim4::new(&[idx_cpu.len() as u64, 1, 1, 1]));

    let dictionary_cpu:Vec<i32> = vec![134,134,213,216,216,321,326,326,326,431,431,553];
	let mut dictionary = arrayfire::Array::new(&dictionary_cpu, arrayfire::Dim4::new(&[dictionary_cpu.len() as u64, 1, 1, 1]));


    let valsel = RayBNN_Sparse::Util::Convert::remap_rows(&dictionary, &idx,1000);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![0,0,1,2,2,3,4,4,4,5,5,6];

    assert_eq!(valsel_act, valsel_cpu);


















	let test_cpu: Vec<f64> = vec![0.3, 0.1, 0.5, 0.9, 0.8, 0.7, -0.1,             2.3, 2.1, 2.5, 2.9, 2.8, 2.7, -2.1,              5.3, 5.1, 5.5, 5.9, 5.8, 5.7, -5.1,           9.3, 9.1, 9.5, 9.9, 9.8, 9.7, -9.1,    ];
	let mut test_arr = arrayfire::Array::new(&test_cpu, arrayfire::Dim4::new(&[7, 4, 1, 1]));

	test_arr = arrayfire::transpose(&test_arr, false);



	let idx_cpu: Vec<u32> = vec![6,3,5,1,0,      1,4,2,4,1,     2,0,3,1,4,    5,2,6,1,2];
	let mut idx_arr = arrayfire::Array::new(&idx_cpu, arrayfire::Dim4::new(&[5, 4, 1, 1]));

	idx_arr = arrayfire::transpose(&idx_arr, false);


	let result = RayBNN_Sparse::Util::Search::parallel_lookup(
		0,
		1,
	
		&idx_arr,
		&test_arr,
	);



	let actual_cpu: Vec<f64> = vec![ -0.1, 0.9, 0.7,  0.1 ,  0.3,            2.1, 2.8, 2.5, 2.8,2.1,       5.5, 5.3, 5.9, 5.1, 5.8,           9.7,  9.5, -9.1, 9.1, 9.5];
	let mut actual = arrayfire::Array::new(&actual_cpu, arrayfire::Dim4::new(&[5, 4, 1, 1]));

	actual = arrayfire::transpose(&actual, false);


	let mut actual_cpu = vec!(f64::default();actual.elements());

    actual.host(&mut actual_cpu);


    let mut result_cpu = vec!(f64::default();result.elements());

    result.host(&mut result_cpu);

    assert_eq!( actual_cpu, result_cpu);




























	let test_cpu: Vec<f64> = vec![0.3, 0.1, 0.5, 0.9, 0.8, 0.7, -0.1,             2.3, 2.1, 2.5, 2.9, 2.8, 2.7, -2.1,              5.3, 5.1, 5.5, 5.9, 5.8, 5.7, -5.1,           9.3, 9.1, 9.5, 9.9, 9.8, 9.7, -9.1,    ];
	let mut test_arr = arrayfire::Array::new(&test_cpu, arrayfire::Dim4::new(&[7, 4, 1, 1]));

	test_arr = arrayfire::transpose(&test_arr, false);

    test_arr = arrayfire::reorder_v2(&test_arr, 0, 2, Some(vec![1]));


	let idx_cpu: Vec<u32> = vec![6,3,5,1,0,      1,4,2,4,1,     2,0,3,1,4,    5,2,6,1,2];
	let mut idx_arr = arrayfire::Array::new(&idx_cpu, arrayfire::Dim4::new(&[5, 4, 1, 1]));

	idx_arr = arrayfire::transpose(&idx_arr, false);

    idx_arr = arrayfire::reorder_v2(&idx_arr, 0, 2, Some(vec![1]));


	let result = RayBNN_Sparse::Util::Search::parallel_lookup(
		0,
		2,
	
		&idx_arr,
		&test_arr,
	);



	let actual_cpu: Vec<f64> = vec![ -0.1, 0.9, 0.7,  0.1 ,  0.3,            2.1, 2.8, 2.5, 2.8,2.1,       5.5, 5.3, 5.9, 5.1, 5.8,           9.7,  9.5, -9.1, 9.1, 9.5];
	let mut actual = arrayfire::Array::new(&actual_cpu, arrayfire::Dim4::new(&[5, 4, 1, 1]));

	actual = arrayfire::transpose(&actual, false);

    actual = arrayfire::reorder_v2(&actual, 0, 2, Some(vec![1]));


	let mut actual_cpu = vec!(f64::default();actual.elements());

    actual.host(&mut actual_cpu);


    let mut result_cpu = vec!(f64::default();result.elements());

    result.host(&mut result_cpu);

    assert_eq!( actual_cpu, result_cpu);



























    
	let test_cpu: Vec<u32> = vec![3, 1, 5, 9, 8, 7, 6,             23, 21, 25, 29, 28, 27, 26,              53, 51, 55, 59, 58, 57, 56,           93, 91, 95, 99, 98, 97, 96,    ];
	let mut test_arr = arrayfire::Array::new(&test_cpu, arrayfire::Dim4::new(&[7, 4, 1, 1]));

	test_arr = arrayfire::transpose(&test_arr, false);


	let idx_cpu: Vec<u32> = vec![6,3,5,1,0,      1,4,2,4,1,     2,0,3,1,4,    5,2,6,1,2];
	let mut idx_arr = arrayfire::Array::new(&idx_cpu, arrayfire::Dim4::new(&[5, 4, 1, 1]));

	idx_arr = arrayfire::transpose(&idx_arr, false);

	let result =  RayBNN_Sparse::Util::Search::parallel_lookup(
		0,
		1,
	
		&idx_arr,
		&test_arr,
	);



	let actual_cpu: Vec<u32> = vec![ 6, 9, 7,  1 ,  3,            21, 28, 25, 28,21,       55, 53, 59, 51, 58,           97,  95, 96, 91, 95];
	let mut actual = arrayfire::Array::new(&actual_cpu, arrayfire::Dim4::new(&[5, 4, 1, 1]));

	actual = arrayfire::transpose(&actual, false);


	let mut actual_cpu = vec!(u32::default();actual.elements());

    actual.host(&mut actual_cpu);


    let mut result_cpu = vec!(u32::default();result.elements());

    result.host(&mut result_cpu);

    assert_eq!( actual_cpu, result_cpu);


















    let input_cpu:Vec<i32> = vec![521, 223, 521, 231, 443, 443, 831, 21,  521, 231, 443, 21, 521, 521, 521, 443, 521];
    let mut input = arrayfire::Array::new(&input_cpu, arrayfire::Dim4::new(&[input_cpu.len() as u64, 1, 1, 1]));


    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let mut bins = arrayfire::constant::<i32>(0,temp_dims);
	let mut counts = arrayfire::constant::<i32>(0,temp_dims);

	RayBNN_Sparse::Util::Search::integer_histogram(
        &input,
        &mut bins,
        &mut counts
    );

    let mut bins_cpu = vec!(i32::default();bins.elements());
    bins.host(&mut bins_cpu);

    let mut counts_cpu = vec!(i32::default();counts.elements());
    counts.host(&mut counts_cpu);

    let mut bins_act:Vec<i32> = vec![ 21, 223, 231, 443, 521, 831];
    let mut counts_act:Vec<i32> = vec![2,   1,   2,   4,   7,   1];

    assert_eq!(bins_cpu, bins_act );

    assert_eq!(counts_cpu, counts_act );
}
