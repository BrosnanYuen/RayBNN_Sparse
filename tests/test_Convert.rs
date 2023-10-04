
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


















    let WValues_cpu: [f64; 12] = [4.1, 1.7, -0.9, 0.3, -2.0, 5.0, -1.0, 0.42, -0.2, 0.1, 3.1, 10.0];
	let mut WValues = arrayfire::Array::new(&WValues_cpu, arrayfire::Dim4::new(&[12, 1, 1, 1]));


    let WRowIdxCSR_cpu: [i32; 8] = [0,0,0,2,4,6,9,12];
	let mut WRowIdxCSR = arrayfire::Array::new(&WRowIdxCSR_cpu, arrayfire::Dim4::new(&[8, 1, 1, 1]));


    let WColIdx_cpu: [i32; 12] = [0,1,0,1,0,1,2,3,4,2,3,4];
	let mut WColIdx = arrayfire::Array::new(&WColIdx_cpu, arrayfire::Dim4::new(&[12, 1, 1, 1]));



    let W = arrayfire::sparse::<f64>(
		7,
		7,
		&WValues,
		&WRowIdxCSR,
		&WColIdx,
		arrayfire::SparseFormat::CSR);

    let Wtemp = arrayfire::sparse_to_dense(&W);




    let mut WRowIdxCOO = RayBNN_Sparse::Util::Convert::CSR_to_COO(&WRowIdxCSR);



    let WRowIdxCOO_cpuact:Vec<i32> = vec![2,2,3,3,4,4,5,5,5,6,6,6];

    let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);


    assert_eq!(WRowIdxCOO_cpu, WRowIdxCOO_cpuact);














    let WRowIdxCSR_cpu:Vec<i32> = vec![0,5,5,6,10,10,10,18,20];
	let mut WRowIdxCSR = arrayfire::Array::new(&WRowIdxCSR_cpu, arrayfire::Dim4::new(&[WRowIdxCSR_cpu.len() as u64, 1, 1, 1]));



    let mut WRowIdxCOO = RayBNN_Sparse::Util::Convert::CSR_to_COO(&WRowIdxCSR);


    let WRowIdxCOO_cpuact:Vec<i32> = vec![0,0,0,0,0,2,3,3,3,3,6,6,6,6,6,6,6,6,7,7];

    let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);


    assert_eq!(WRowIdxCOO_cpu, WRowIdxCOO_cpuact);

























    let WRowIdxCOO_cpu:Vec<i32> = vec![23,23,31,34, 40,40,40];
	let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[WRowIdxCOO_cpu.len() as u64, 1, 1, 1]));


    let WColIdx_cpu:Vec<i32> = vec![120,140,315,421, 64,71,91];
	let mut WColIdx = arrayfire::Array::new(&WColIdx_cpu, arrayfire::Dim4::new(&[WColIdx_cpu.len() as u64, 1, 1, 1]));



    let global_idx = RayBNN_Sparse::Util::Convert::get_global_weight_idx(
        2000, 
        &WRowIdxCOO, 
        &WColIdx
    );

    let global_idx_act:Vec<u64> = vec![46120, 46140, 62315, 68421, 80064, 80071, 80091];

    let mut global_idx_cpu = vec!(u64::default();global_idx.elements());
    global_idx.host(&mut global_idx_cpu);

    assert_eq!(global_idx_act, global_idx_cpu);













    let WRowIdxCOO_cpu:Vec<i32> = vec![23,23,31,34, 40,40,40];
	let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[WRowIdxCOO_cpu.len() as u64, 1, 1, 1]));


    let WColIdx_cpu:Vec<i32> = vec![120,140,315,421, 64,71,91];
	let mut WColIdx = arrayfire::Array::new(&WColIdx_cpu, arrayfire::Dim4::new(&[WColIdx_cpu.len() as u64, 1, 1, 1]));



    let global_idx = RayBNN_Sparse::Util::Convert::get_global_weight_idx2(
        2000, 
        &WRowIdxCOO, 
        &WColIdx
    );

    let global_idx_act:Vec<u64> = vec![46120, 46140, 62315, 68421, 80064, 80071, 80091];

    let mut global_idx_cpu = vec!(u64::default();global_idx.elements());
    global_idx.host(&mut global_idx_cpu);

    assert_eq!(global_idx_act, global_idx_cpu);



}
