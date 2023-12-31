
use arrayfire;
use RayBNN_Sparse;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_search() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);






    let WRowIdxCOO_cpu:Vec<i32> = vec![2,2,3,3,4,4,5,5,5,6,6,6];
	let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[12, 1, 1, 1]));


    let idxsel_cpu: [i32; 2] = [5,6];
	let mut idxsel = arrayfire::Array::new(&idxsel_cpu, arrayfire::Dim4::new(&[2, 1, 1, 1]));

    let valsel = RayBNN_Sparse::Util::Search::COO_find(&WRowIdxCOO,&idxsel);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![6,7,8,9,10,11];


    assert_eq!(valsel_act, valsel_cpu);









    let idxsel_cpu: [i32; 3] = [2,3,5];
	let mut idxsel = arrayfire::Array::new(&idxsel_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));

    let valsel = RayBNN_Sparse::Util::Search::COO_find(&WRowIdxCOO,&idxsel);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![0,1,2,3,6,7,8];


    assert_eq!(valsel_act, valsel_cpu);













    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,0,0,2,3,3,5,6,6];
    let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[10, 1, 1, 1]));


    let idxsel_cpu: [i32; 3] = [0,3,5];
	let mut idxsel = arrayfire::Array::new(&idxsel_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));

    let valsel = RayBNN_Sparse::Util::Search::COO_find(&WRowIdxCOO,&idxsel);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![0,1,2,3,5,6,7];


    assert_eq!(valsel_act, valsel_cpu);







    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,0,0,2,3,3,5,6,6,7,7,7,7];
    let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[14, 1, 1, 1]));


    let idxsel_cpu: [i32; 3] = [0,5,7];
	let mut idxsel = arrayfire::Array::new(&idxsel_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));

    let valsel = RayBNN_Sparse::Util::Search::COO_find(&WRowIdxCOO,&idxsel);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![0,1,2,3,7,10,11,12,13];


    assert_eq!(valsel_act, valsel_cpu);






















    let WRowIdxCOO_cpu:Vec<i32> = vec![2,2,3,3,4,4,5,5,5,6,6,6];
	let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[12, 1, 1, 1]));


    let idxsel_cpu: [i32; 2] = [5,6];
	let mut idxsel = arrayfire::Array::new(&idxsel_cpu, arrayfire::Dim4::new(&[2, 1, 1, 1]));

    let valsel = RayBNN_Sparse::Util::Search::COO_batch_find(&WRowIdxCOO,&idxsel,20);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![6,7,8,9,10,11];


    assert_eq!(valsel_act, valsel_cpu);
















    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,0,0,2,3,3,5,6,6,7,7,7,7];
    let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[14, 1, 1, 1]));


    let idxsel_cpu: [i32; 3] = [0,5,7];
	let mut idxsel = arrayfire::Array::new(&idxsel_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));

    let valsel = RayBNN_Sparse::Util::Search::COO_batch_find(&WRowIdxCOO,&idxsel,30);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![0,1,2,3,7,10,11,12,13];


    assert_eq!(valsel_act, valsel_cpu);















    let WRowIdxCOO_cpu:Vec<i32> = vec![0,0,0,1,2,2,3,3,3,4,6,6,8,8,9,9,10,11,11,15,15,17,20,20,20,21,21,21,23,23];
    let mut WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[WRowIdxCOO_cpu.len() as u64, 1, 1, 1]));


    let idxsel_cpu:Vec<i32> = vec![1,2,5,6,9,10,15,18,19,20,23];
	let mut idxsel = arrayfire::Array::new(&idxsel_cpu, arrayfire::Dim4::new(&[idxsel_cpu.len()  as u64, 1, 1, 1]));

    let valsel = RayBNN_Sparse::Util::Search::COO_batch_find(&WRowIdxCOO,&idxsel,4);
    let mut valsel_cpu = vec!(i32::default();valsel.elements());
    valsel.host(&mut valsel_cpu);


    let valsel_act:Vec<i32> = vec![3,4,5,10,11,14,15,16,19,20,22,23,24,28,29];


    assert_eq!(valsel_act, valsel_cpu);








    let in_idx_cpu:Vec<i32> = vec![0,0,0,2,4,4,5,5,5,5];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[10, 1, 1, 1]));


    let out_idx = RayBNN_Sparse::Util::Search::find_unique(
        &in_idx,
        6
    );

    let out_idx_act:Vec<i32> = vec![0,2,4,5];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);












    let in_idx_cpu:Vec<i32> = vec![6,3,0,1,5,3,7,0,11,1,5,5,5,1];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[14, 1, 1, 1]));


    let out_idx = RayBNN_Sparse::Util::Search::find_unique(
            &in_idx,
            20
    	);

    let out_idx_act:Vec<i32> = vec![0,1,3,5,6,7,11];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);











    let in_idx_cpu:Vec<i32> = vec![1,4,6,7,9,2,4,5,7,9,1];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[11, 1, 1, 1]));


    let out_idx = RayBNN_Sparse::Util::Search::find_unique(
            &in_idx,
            20
    	);

    let out_idx_act:Vec<i32> = vec![1,2,4,5,6,7,9];

    let mut out_idx_cpu = vec!(i32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);









    let in_idx_cpu:Vec<u32> = vec![1,4,6,7,9,2,4,5,7,9,1];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[11, 1, 1, 1]));


    let out_idx = RayBNN_Sparse::Util::Search::find_unique(
            &in_idx,
            20
    	);

    let out_idx_act:Vec<u32> = vec![1,2,4,5,6,7,9];

    let mut out_idx_cpu = vec!(u32::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);












    let in_idx_cpu:Vec<i64> = vec![6,3,0,1,5,3,7,0,11,1,5,5,5,1];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[14, 1, 1, 1]));


    let out_idx = RayBNN_Sparse::Util::Search::find_unique(
            &in_idx,
            20
    	);

    let out_idx_act:Vec<i64> = vec![0,1,3,5,6,7,11];

    let mut out_idx_cpu = vec!(i64::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);















    let in_idx_cpu:Vec<u64> = vec![6,3,0,1,5,3,7,0,11,1,5,5,5,1];
	let mut in_idx = arrayfire::Array::new(&in_idx_cpu, arrayfire::Dim4::new(&[14, 1, 1, 1]));


    let out_idx = RayBNN_Sparse::Util::Search::find_unique(
            &in_idx,
            20
    	);

    let out_idx_act:Vec<u64> = vec![0,1,3,5,6,7,11];

    let mut out_idx_cpu = vec!(u64::default();out_idx.elements());
    out_idx.host(&mut out_idx_cpu);

    assert_eq!(out_idx_act, out_idx_cpu);



}
