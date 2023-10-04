
use arrayfire;
use RayBNN_Sparse;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_Remove3() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);


    let WValues_cpu:Vec<f32> = vec![9.0,-2.8,5.1,-0.3,4.8,4.1, 1.7, -0.9, 0.04, -2.0, 5.0, -1.0, 0.42, -0.2, 0.1, 3.1, 10.2, 5.9];
	let mut WValues = arrayfire::Array::new(&WValues_cpu, arrayfire::Dim4::new(&[18, 1, 1, 1]));


    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,1,1,1,2,2,3,3,4,4,5,5,5,6,6,6,6];
	let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[18, 1, 1, 1]));


    let WColIdx_cpu:Vec<i32> = vec![2,3,4,5,6,0,1,0,1,0,1,2,3,4,2,3,4,5];
	let mut WColIdx = arrayfire::Array::new(&WColIdx_cpu, arrayfire::Dim4::new(&[18, 1, 1, 1]));


    
    RayBNN_Sparse::Util::Remove::delete_smallest_weights::<f32>(
        &mut WValues,
        &mut WRowIdxCOO,
        &mut WColIdx,
        3
    );





    let WValues_act:Vec<f32> = vec![9.0,-2.8,5.1,-0.3,4.8,4.1, 1.7, -0.9, -2.0, 5.0, -1.0, 0.42, 3.1, 10.2, 5.9];


    let WRowIdxCOO_act:Vec<i32> = vec![0,0,1,1,1,2,2,3,4,4,5,5,6,6,6];


    let WColIdx_act:Vec<i32> = vec![2,3,4,5,6,0,1,0,0,1,2,3,3,4,5];






    let mut WValues_cpu = vec!(f32::default();WValues.elements());
    WValues.host(&mut WValues_cpu);

    assert_eq!(WValues_act, WValues_cpu);








    let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

    assert_eq!(WRowIdxCOO_act, WRowIdxCOO_cpu);







    let mut WColIdx_cpu = vec!(i32::default();WColIdx.elements());
    WColIdx.host(&mut WColIdx_cpu);

    assert_eq!(WColIdx_act, WColIdx_cpu);




}
