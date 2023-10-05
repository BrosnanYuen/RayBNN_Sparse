
use arrayfire;
use RayBNN_Sparse;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_search() {

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






}
