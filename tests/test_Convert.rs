
use arrayfire;
use RayBNN_Sparse;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_convert() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);



    let WRowIdxCOO_cpu:Vec<i32> = vec![2,2,3,3,4,4,5,5,5,6,6,6];
    let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[12, 1, 1, 1]));


    let mut WRowIdxCSR = RayBNN_Sparse::Util::Convert::COO_to_CSR(&WRowIdxCOO,7);
    let mut WRowIdxCSR_cpu = vec!(i32::default();WRowIdxCSR.elements());
    WRowIdxCSR.host(&mut WRowIdxCSR_cpu);

    let WRowIdxCSR_act:Vec<i32> = vec![0,0,0,2,4,6,9,12];


    assert_eq!(WRowIdxCSR_cpu, WRowIdxCSR_act);











    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,0,0,2,3,3,5,6,6];
    let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[10, 1, 1, 1]));


    let mut WRowIdxCSR = RayBNN_Sparse::Util::Convert::COO_to_CSR(&WRowIdxCOO,7);
    let mut WRowIdxCSR_cpu = vec!(i32::default();WRowIdxCSR.elements());
    WRowIdxCSR.host(&mut WRowIdxCSR_cpu);

    let WRowIdxCSR_act:Vec<i32> = vec![0,4,4,5,7,7,8,10];


    assert_eq!(WRowIdxCSR_cpu, WRowIdxCSR_act);







    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,0,1,1,5,5];
    let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[7, 1, 1, 1]));


    let mut WRowIdxCSR = RayBNN_Sparse::Util::Convert::COO_to_CSR(&WRowIdxCOO,7);
    let mut WRowIdxCSR_cpu = vec!(i32::default();WRowIdxCSR.elements());
    WRowIdxCSR.host(&mut WRowIdxCSR_cpu);

    let WRowIdxCSR_act:Vec<i32> = vec![0,3,5,5,5,5,7,7];


    assert_eq!(WRowIdxCSR_cpu, WRowIdxCSR_act);











    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,0,1,1];
    let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[5, 1, 1, 1]));


    let mut WRowIdxCSR = RayBNN_Sparse::Util::Convert::COO_to_CSR(&WRowIdxCOO,7);
    let mut WRowIdxCSR_cpu = vec!(i32::default();WRowIdxCSR.elements());
    WRowIdxCSR.host(&mut WRowIdxCSR_cpu);

    let WRowIdxCSR_act:Vec<i32> = vec![0,3,5,5,5,5,5,5];


    assert_eq!(WRowIdxCSR_cpu, WRowIdxCSR_act);



}
